import hashlib
import json
import pickle
import re
from pathlib import Path


def _slugify(value):
    text = str(value).strip().lower()
    slug = re.sub(r'[^a-z0-9]+', '-', text).strip('-')
    return slug or 'system'


def _cfg_hash(cfg):
    cfg_text = json.dumps(cfg, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha1(cfg_text.encode('utf-8')).hexdigest()[:12]


def build_gt_cache_path(
    cache_dir,
    system_name,
    cfg,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    system_tag = _slugify(system_name)
    filename = f'{system_tag}_cfg_{_cfg_hash(cfg)}.pkl'
    return cache_dir / filename


def load_gt_cache(cache_path):
    cache_path = Path(cache_path)
    with cache_path.open('rb') as handle:
        return pickle.load(handle)


def save_gt_cache(cache_path, gt_reach_regions):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open('wb') as handle:
        pickle.dump(gt_reach_regions, handle, protocol=pickle.HIGHEST_PROTOCOL)
