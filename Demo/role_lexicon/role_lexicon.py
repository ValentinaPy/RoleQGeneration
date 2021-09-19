import itertools
from typing import Tuple, Union, List, Optional
from dataclasses import dataclass
import pandas as pd
from jsonlines import jsonlines

from Demo.role_lexicon.common_types import Role, Predicate


class RoleLexicon:
    def __init__(self, entries):
        entries = [Role(predicate=e.get('lemma') or e.get("predicate_lemma"),
                        sense_id=f"{e['sense_id']:02d}",
                        pos=RoleLexicon.normalize_pos(e['pos']),
                        role_type=e['role_type'],
                        role_desc=e['role_desc'], role_set_desc=e['roleset_desc'])
                   for e in entries]
        roleset_map = {}
        all_rolesets_map = {}
        entries = sorted(entries, key=RoleLexicon.role_sort_order)
        for key_items, roleset in itertools.groupby(entries, key=RoleLexicon.roleset_sort_order):
            key = self.to_key(*key_items)
            roleset_map[key] = list(roleset)

        for key_items, roleset in itertools.groupby(entries, key=RoleLexicon.all_rolesets_sort_order):
            key = '.'.join(key_items)
            all_rolesets_map[key] = list(roleset)

        self.roleset_map = roleset_map
        self.all_rolesets_map = all_rolesets_map

    def __getitem__(self, item) -> Union[Role, List[Role]]:
        if len(item) == 4:
            predicate_lemma, sense, pos, role = item[:4]
            return self.get_role(predicate_lemma, sense, pos, role)
        if len(item) == 3:
            predicate_lemma, sense, pos = item[:3]
            return self.get_roleset(predicate_lemma, sense, pos)
        if len(item) == 2:
            predicate, role = item[:2]
            return self.get_role(predicate.lemma, predicate.sense_id, predicate.pos, role)

    def get_roleset(self, predicate: Predicate) -> Optional[List[Role]]:
        return self.get_roleset(predicate.lemma, predicate.sense_id, predicate.pos)

    def get_roleset(self, predicate: str, sense_id: str, pos: str) -> Optional[List[Role]]:
        key = self.to_key(predicate, sense_id, pos)
        return self.roleset_map.get(key)

    def get_all_rolesets(self, predicate: str, pos: str) -> Optional[List[Role]]:
        key = f"{predicate}.{pos}"
        return self.all_rolesets_map.get(key)

    def get_role(self, predicate: str, sense_id: str, pos: str, role: str) -> Optional[Role]:
        roleset = self.get_roleset(predicate, sense_id, pos)
        if not roleset:
            return None

        roles = [r for r in roleset if r.role_type == role]
        return next(iter(roles), None)

    @staticmethod
    def normalize_pos(pos: str) -> str:
        return pos[0].lower()

    def to_key(self, *args) -> str:
        predicate_lemma, sense_id, pos = args[:3]
        pos = RoleLexicon.normalize_pos(pos)
        if isinstance(sense_id, int):
            sense_id = f"{sense_id:02d}"
        return f"{predicate_lemma}.{sense_id}{pos}"

    @staticmethod
    def role_sort_order(entry: Role):
        e = entry
        return e.predicate, e.sense_id, e.pos, e.role_type

    @staticmethod
    def roleset_sort_order(entry: Role):
        e = entry
        return e.predicate, e.sense_id, e.pos

    @staticmethod
    def all_rolesets_sort_order(entry: Role):
        e = entry
        return e.predicate, e.pos

    @classmethod
    def from_file(cls, lexicon_path: str) -> 'RoleLexicon':
        if lexicon_path.endswith(".tsv"):
            df = pd.read_csv(lexicon_path, sep="\t")
            roles = df.to_dict(orient="records")
            return cls(roles)
        if lexicon_path.endswith(".jsonl"):
            list(jsonlines.open(lexicon_path))
            return cls()
        else:
            raise NotImplementedError(f"unrecognized extension: {lexicon_path}")

