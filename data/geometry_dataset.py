"""Structured-acquisition SoS dataset.

Each cache record contains per-speed/per-event complex DAS images, receive
sub-apertures, target maps, physical event geometry and validity masks.  Event
masking is applied here rather than by zeroing channels after a fixed tensor
layout, so the encoder knows which transmit angles are physically absent.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class StructuredSoSDataset(Dataset):
    def __init__(self, cache_dir, split='train', event_dropout_p=0.,
                 min_events=6, lateral_mirror=False, seed=0, limit=0):
        self.cache_dir = Path(cache_dir)
        manifest = json.loads((self.cache_dir/'manifest.json').read_text())
        self.manifest = manifest
        records = [r for r in manifest['records']
                   if split == 'all' or r.get('split') == split]
        if limit:
            # ``limit`` counts independent source examples, not cache rows:
            # clean and RF-augmented variants of one source share source_id.
            selected, sources = [], set()
            for record in records:
                source = record.get('source_id', record['id'])
                if len(sources) < int(limit) or source in sources:
                    selected.append(record)
                    sources.add(source)
            records = selected
        if not records:
            raise FileNotFoundError(f'No {split} records in {self.cache_dir}')
        self.records = records
        self.split = split
        self.event_dropout_p = float(event_dropout_p) if split == 'train' else 0.
        self.min_events = int(min_events)
        self.lateral_mirror = bool(lateral_mirror) and split == 'train'
        self.seed = int(seed)
        if self.event_dropout_p < 0 or self.event_dropout_p >= 1:
            raise ValueError('event_dropout_p must be in [0,1)')

    def __len__(self):
        return len(self.records)

    def _load(self, record):
        path = self.cache_dir/record.get('cache_file', record['id'] + '.npz')
        with np.load(path, allow_pickle=False) as d:
            item = {k: d[k] for k in d.files}
        return item

    def _drop_events(self, mask, idx):
        mask = np.asarray(mask, bool).copy()
        if self.event_dropout_p <= 0:
            return mask
        rng = np.random.default_rng(self.seed + 104729*idx + int(np.random.randint(1 << 30)))
        active = np.flatnonzero(mask)
        drop = rng.random(active.size) < self.event_dropout_p
        if active.size - int(drop.sum()) < self.min_events:
            keep = rng.choice(active, size=min(self.min_events, active.size), replace=False)
            mask[:] = False
            mask[keep] = True
        else:
            mask[active[drop]] = False
        return mask

    def _mirror_permutation(self, event_geom):
        angles = np.arctan2(event_geom[:, 0], event_geom[:, 1])
        perm = []
        for angle in angles:
            candidates = np.where(np.isclose(angles, -angle, atol=1e-5))[0]
            if candidates.size != 1:
                raise ValueError('Lateral mirror requires a symmetric event-angle set')
            perm.append(int(candidates[0]))
        return np.asarray(perm, dtype=np.int64)

    def __getitem__(self, idx):
        record = self.records[idx]
        item = self._load(record)
        speed_events = np.asarray(item['speed_events'], np.complex64)
        subap = np.asarray(item['subap'], np.complex64)
        subap_events = (np.asarray(item['subap_events'], np.complex64)
                        if 'subap_events' in item else None)
        event_geom = np.asarray(item['event_geom'], np.float32)
        subap_geom = np.asarray(item['subap_geom'], np.float32)
        global_geom = np.asarray(item['global_geom'], np.float32)
        event_mask = np.asarray(item['event_mask'], bool)
        subap_mask = np.asarray(item['subap_mask'], bool)
        event_indices = np.asarray(item.get('event_indices', np.arange(speed_events.shape[1])), np.int64)
        c_gt = np.asarray(item['c_gt'], np.float32)
        u_gt = np.asarray(item['u_gt'], np.float32)[None]

        event_mask = self._drop_events(event_mask, idx)
        if self.lateral_mirror and np.random.rand() < .5:
            perm = self._mirror_permutation(event_geom)
            speed_events = speed_events[:, perm, ::-1, :]
            event_mask = event_mask[perm]
            subap = subap[::-1, ::-1, :]
            if subap_events is not None:
                subap_events = subap_events[perm, ::-1, ::-1, :]
            subap_mask = subap_mask[::-1]
            c_gt = c_gt[::-1, :]
            u_gt = u_gt[:, ::-1, :]

        condition = {
            'speed_events': torch.from_numpy(np.ascontiguousarray(speed_events)),
            'subap': torch.from_numpy(np.ascontiguousarray(subap)),
            **({'subap_events': torch.from_numpy(np.ascontiguousarray(subap_events))}
               if subap_events is not None else {}),
            'event_geom': torch.from_numpy(np.ascontiguousarray(event_geom)),
            'subap_geom': torch.from_numpy(np.ascontiguousarray(subap_geom)),
            'global_geom': torch.from_numpy(np.ascontiguousarray(global_geom)),
            'event_mask': torch.from_numpy(np.ascontiguousarray(event_mask)),
            'subap_mask': torch.from_numpy(np.ascontiguousarray(subap_mask)),
        }
        return {
            'condition': condition,
            'u_gt': torch.from_numpy(np.ascontiguousarray(u_gt)),
            'c_gt': torch.from_numpy(np.ascontiguousarray(c_gt[None])),
            'input_event_indices': torch.from_numpy(event_indices[event_mask]),
            'name': record['id'],
            'source_id': record.get('source_id', record['id']),
        }


def geometry_collate(items):
    """Pad variable event counts and preserve masks in a batch."""
    b = len(items)
    s, h, w = items[0]['condition']['speed_events'].shape[0], *items[0]['condition']['speed_events'].shape[-2:]
    a_max = max(i['condition']['speed_events'].shape[1] for i in items)
    event_dim = items[0]['condition']['event_geom'].shape[-1]
    speed_events = torch.zeros(b, s, a_max, h, w, dtype=torch.complex64)
    have_subap_events = all('subap_events' in i['condition'] for i in items)
    if any(('subap_events' in i['condition']) != have_subap_events for i in items):
        raise ValueError('Mixed legacy/event-resolved subap cache records in one batch')
    k_subap = items[0]['condition']['subap'].shape[0]
    subap_events = (torch.zeros(b, a_max, k_subap, h, w, dtype=torch.complex64)
                    if have_subap_events else None)
    event_geom = torch.zeros(b, a_max, event_dim)
    event_mask = torch.zeros(b, a_max, dtype=torch.bool)
    for bi, item in enumerate(items):
        a = item['condition']['speed_events'].shape[1]
        speed_events[bi, :, :a] = item['condition']['speed_events']
        if subap_events is not None:
            subap_events[bi, :a] = item['condition']['subap_events']
        event_geom[bi, :a] = item['condition']['event_geom']
        event_mask[bi, :a] = item['condition']['event_mask']
    condition = {
        'speed_events': speed_events,
        'event_geom': event_geom,
        'event_mask': event_mask,
        'subap': torch.stack([i['condition']['subap'] for i in items]),
        **({'subap_events': subap_events} if subap_events is not None else {}),
        'subap_geom': torch.stack([i['condition']['subap_geom'] for i in items]),
        'subap_mask': torch.stack([i['condition']['subap_mask'] for i in items]),
        'global_geom': torch.stack([i['condition']['global_geom'] for i in items]),
    }
    return {
        'condition': condition,
        'u_gt': torch.stack([i['u_gt'] for i in items]),
        'c_gt': torch.stack([i['c_gt'] for i in items]),
        'input_event_indices': [i['input_event_indices'] for i in items],
        'name': [i['name'] for i in items],
        'source_id': [i['source_id'] for i in items],
    }
