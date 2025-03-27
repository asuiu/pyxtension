# Author: ASU --<andrei.suiu@gmail.com>
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Dict, List
from unittest import TestCase

import pydantic
from pydantic import ConfigDict, field_serializer, field_validator
from streamerate import slist
from tsx import TS

from pyxtension import PydanticCoercingValidated
from pyxtension.models.v2 import ExtModel, ImmutableExtModel


class CustomFloatWithVal(float, PydanticCoercingValidated):
    pass


class FrozenMinion(ExtModel):
    name: str
    birth_date: TS

    @field_serializer("birth_date", when_used="json")
    def serialize_brith_date(self, ts: TS) -> str:
        return ts.as_iso


class MinionWithFamily(FrozenMinion):
    members: List[FrozenMinion]


class MinionWithWallet(MinionWithFamily):
    wallet: CustomFloatWithVal

    @field_serializer("wallet", when_used="json")
    def serialize_wallet(self, cf: CustomFloatWithVal) -> float:
        return cf + 0.5

    @field_validator("wallet", mode="before")
    def decrease_wallet(cls, v):
        if not isinstance(v, CustomFloatWithVal):
            v = CustomFloatWithVal(v - 0.2)
        return v


class TestExtModel(TestCase):
    def test_to_from_json_nominal(self):
        class CustomFloat(float, PydanticCoercingValidated):
            pass

        class A(ExtModel):
            ts: int
            cf: CustomFloat

        a = A(ts=1, cf=CustomFloat(1.0))
        serialized_json = a.model_dump_json()
        v1_json = a.json()
        self.assertEqual(v1_json, serialized_json)
        expected = '{"ts":1,"cf":1.0}'
        self.assertEqual(expected, serialized_json)
        d = json.loads(serialized_json)
        new_a = A(**d)
        self.assertEqual(a, new_a)
        self.assertIsInstance(new_a.cf, CustomFloat)

    def test_to_from_json(self):
        class CustomFloat(float, PydanticCoercingValidated):
            pass

        class A(ExtModel):
            ts: TS

            @field_serializer("ts", when_used="json")
            def serialize_ts(self, ts: TS) -> str:
                return ts.as_iso_date_basic

        class B(A):
            cf: CustomFloat
            a_list: List[A]
            a_slist: slist[A]
            a_dict: Dict[int, A]

            model_config = ConfigDict(json_encoders={CustomFloat: lambda cf: cf + 0.5})

            @field_serializer("cf", when_used="json")
            def serialize_cf(self, cf: CustomFloat) -> float:
                return float(cf) + 0.5

        ts = TS("2023-04-12T00:00:00Z")
        a = A(ts=ts)
        serialized_json_a = a.model_dump_json()
        v1_json = a.json()
        self.assertEqual(v1_json, serialized_json_a)
        self.assertEqual(serialized_json_a, '{"ts":"20230412"}')

        b = B(ts=ts, cf=1.0, a_list=[a], a_slist=slist([a]), a_dict={1: a})
        serialized_json = b.model_dump_json()
        expected = '{"ts":"20230412","cf":1.5,"a_list":[{"ts":"20230412"}],"a_slist":[{"ts":"20230412"}],"a_dict":{"1":{"ts":"20230412"}}}'
        self.assertEqual(expected, serialized_json)
        d = json.loads(serialized_json)
        new_a = B(**d)
        self.assertIsInstance(new_a.ts, TS)
        self.assertIsInstance(new_a.cf, CustomFloat)
        self.assertEqual(new_a.cf, CustomFloat(1.5))
        self.assertIsInstance(new_a.a_list, list)
        self.assertIsInstance(new_a.a_list[0], A)
        self.assertIsInstance(new_a.a_list[0].ts, TS)

    def test_generic_validator(self):
        class CustomFloatWithVal(float, PydanticCoercingValidated):
            pass

        class WithValidatedCF(ExtModel):
            cf: CustomFloatWithVal

        c = WithValidatedCF(cf=1.0)
        self.assertIsInstance(c.cf, CustomFloatWithVal)

    def test_model_with_nested(self):
        m = MinionWithWallet(
            name="Joe",
            birth_date=TS("2001-01-01"),
            members=[FrozenMinion(name="Sara", birth_date=TS("2002-02-02")), FrozenMinion(name="Garry", birth_date=TS("2003-03-03"))],
            wallet=CustomFloatWithVal(1.0),
        )
        self.assertEqual(m.wallet, CustomFloatWithVal(1.0))
        serialized = m.model_dump_json()
        d = json.loads(serialized)
        expected = (
            '{"name":"Joe",'
            '"birth_date":"2001-01-01T00:00:00Z",'
            '"members":[{"name":"Sara","birth_date":"2002-02-02T00:00:00Z"},{"name":"Garry","birth_date":"2003-03-03T00:00:00Z"}],'
            '"wallet":1.5}'
        )
        self.assertEqual(expected, serialized)
        new_m = MinionWithWallet(**d)
        self.assertIsInstance(new_m.wallet, CustomFloatWithVal)
        self.assertEqual(new_m.wallet, CustomFloatWithVal(1.3))  # the custom validator is expecting to decrease the wallet by 0.2
        self.assertIsInstance(new_m.birth_date, TS)
        self.assertIsInstance(new_m.members[0].birth_date, TS)
        self.assertIsInstance(new_m.members[0], FrozenMinion)

    def test_immutable_with_not_frozen_dataclass(self):
        @dataclass
        class A(ImmutableExtModel):
            i: int

        with self.assertRaises(pydantic.ValidationError):
            _ = A(i=1)

    def test_mutable_with_frozen_dataclass(self):
        @dataclass(frozen=True)
        class A(ExtModel):
            i: int

        a = A(i=1)
        with self.assertRaises(AttributeError):
            a.i = 2

    def test_extmodel_with_dataclass(self):
        @dataclass
        class A(ExtModel):
            i: int

        with self.assertRaises(AttributeError):
            _ = A(i=1)

    def test_pydantic_ds(self):
        @pydantic.v1.dataclasses.dataclass(frozen=True)
        class Minion(ExtModel):
            name: str

        @pydantic.v1.dataclasses.dataclass(frozen=True)
        class Boss(ExtModel):
            minions: List[Minion]

        boss = Boss(minions=[Minion(name="evil minion"), Minion(name="very evil minion")])
        expected_json = '{"minions": [{"name": "evil minion"}, {"name": "very evil minion"}]}'  # noqa: F841
        j = boss.json()
        # ToDo: next assert doesn't pass as it adds "__pydantic_initialised__": true to Minion. Investigate.
        # self.assertEqual(j, expected_json)
        d = json.loads(j)
        new_boss = Boss(**d)
        self.assertIsInstance(new_boss.minions[0], Minion)
        self.assertIsInstance(new_boss.minions, List)


class TestImmutableExtModel(TestCase):
    def test_basic_immutability(self):
        class TestModel(ImmutableExtModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)
        with self.assertRaises(pydantic.ValidationError):
            model.name = "new_name"

        with self.assertRaises(pydantic.ValidationError):
            model.value = 100

    def test_nested_immutability(self):
        class NestedModel(ImmutableExtModel):
            data: dict

        model = NestedModel(data={"key": "value"})
        with self.assertRaises(pydantic.ValidationError):
            model.data = {"new": "data"}

        # The nested dict should NOT be immutable
        model.data["key"] = "new_value"

    def test_json_serialization(self):
        class JsonModel(ImmutableExtModel):
            name: str
            value: int

        model = JsonModel(name="test", value=42)
        json_str = model.model_dump_json()
        expected = '{"name":"test","value":42}'
        self.assertEqual(expected, json_str)

        # Test deserialization
        loaded = JsonModel.parse_raw(json_str)
        self.assertEqual(loaded, model)

    def test_with_slists(self):
        class ShallowListing(ImmutableExtModel):
            """
            :param objects: list of object names, as PurePosixPath
            :param prefixes: list of prefixes (equivalent to directories on FileSystems) as strings, ending with "/"
            """

            objects: slist[PurePosixPath]
            prefixes: slist[str]

        model = ShallowListing(objects=list([PurePosixPath("a")]), prefixes=set(["a/", "b/"]))
        self.assertIsInstance(model.objects, slist, f"objects are of type {type(model.objects)}")
        self.assertIsInstance(model.prefixes, slist)
