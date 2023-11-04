# Author: ASU --<andrei.suiu@gmail.com>
import json
from dataclasses import dataclass, FrozenInstanceError
from typing import Optional, List
from unittest import TestCase

import pandas as pd
import pydantic
from pydantic.v1 import BaseModel, validator
from tsx import TS

from pyxtension.models import ExtModel, coercing_field, JsonData, ImmutableExtModel


class CustomFloat(float):
    pass


class CustomFloatWithVal(float):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not isinstance(v, CustomFloatWithVal):
            return CustomFloatWithVal(v)
        return v


class FrozenMinion(ExtModel):
    name: str
    birth_date: TS

    class Config:
        json_encoders = {TS: TS.as_iso.fget}


class MinionWithFamily(FrozenMinion):
    members: List[FrozenMinion]


class MinionWithWallet(MinionWithFamily):
    wallet: CustomFloatWithVal

    class Config:
        json_encoders = {CustomFloat: lambda cf: cf + 0.5, }


class TestExtModel(TestCase):
    def test_to_from_json(self):
        class CustomFloat(float):
            pass

        class A(ExtModel):
            ts: TS

            class Config:
                json_encoders = {
                    TS: TS.as_iso.fget
                }

        class B(A):
            cf: CustomFloat
            a_list: List[A]

            @validator('cf')
            def custom_float_validator(cls, v):
                if not isinstance(v, CustomFloat):
                    return CustomFloat(v)
                return v

            class Config:
                json_encoders = {
                    CustomFloat: lambda cf: cf + 0.5, }

        ts = TS("2023-04-12T00:00:00Z")
        a = B(ts=ts, cf=1., a_list=[A(ts=ts)])
        serialized_json = a.json()
        expected = '{"ts": "2023-04-12T00:00:00Z", "cf": 1.5, "a_list": [{"ts": "2023-04-12T00:00:00Z"}]}'
        self.assertEqual(serialized_json, expected)
        d = json.loads(serialized_json)
        new_a = B(**d)
        self.assertIsInstance(new_a.ts, TS)
        self.assertIsInstance(new_a.cf, CustomFloat)
        self.assertEqual(new_a.cf, CustomFloat(1.5))
        self.assertIsInstance(new_a.a_list, list)
        self.assertIsInstance(new_a.a_list[0], A)
        self.assertIsInstance(new_a.a_list[0].ts, TS)

    def test_generic_validator(self):
        class CustomFloatWithVal(float):
            @classmethod
            def __get_validators__(cls):
                yield cls.validate

            @classmethod
            def validate(cls, v):
                if not isinstance(v, CustomFloatWithVal):
                    return CustomFloatWithVal(v)
                return v

        class WithValidatedCF(ExtModel):
            cf: CustomFloatWithVal

        c = WithValidatedCF(cf=1.)
        self.assertIsInstance(c.cf, CustomFloatWithVal)

    def test_model_with_nested(self):
        m = MinionWithWallet(name="Joe", birth_date=TS("2001-01-01"),
                             members=[FrozenMinion(name="Sara", birth_date=TS("2002-02-02")), FrozenMinion(name="Garry", birth_date=TS("2003-03-03"))],
                             wallet=CustomFloat(1.))
        serialized = m.json()
        d = json.loads(serialized)
        new_m = MinionWithWallet(**d)
        self.assertIsInstance(new_m.wallet, CustomFloatWithVal)
        self.assertIsInstance(new_m.birth_date, TS)
        self.assertIsInstance(new_m.members[0].birth_date, TS)
        self.assertIsInstance(new_m.members[0], FrozenMinion)
        self.assertEqual(m, new_m)

    def test_immutable_with_not_frozen_dataclass(self):
        @dataclass
        class A(ImmutableExtModel):
            i: int

        with self.assertRaises(TypeError):
            a = A(i=1)

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

        boss = Boss(minions=[Minion(name='evil minion'), Minion(name='very evil minion')])
        expected_json = '{"minions": [{"name": "evil minion"}, {"name": "very evil minion"}]}'
        j = boss.json()
        # ToDo: next assert doesn't pass as it adds "__pydantic_initialised__": true to Minion. Investigate.
        # self.assertEqual(j, expected_json)
        d = json.loads(j)
        new_boss = Boss(**d)
        self.assertIsInstance(new_boss.minions[0], Minion)
        self.assertIsInstance(new_boss.minions, List)


class TestJsonData(TestCase):
    @dataclass
    class A(JsonData):
        ts: TS = coercing_field()

        class Config:
            json_encoders = {
                TS: TS.as_iso.fget
            }

    @dataclass
    class B(A):
        cf: CustomFloat = coercing_field()

        class Config:
            json_encoders = {
                CustomFloat: lambda cf: cf + 0.5, }

    def test_custom_json_encoding(self):
        ts = TS("2023-04-12T00:00:00Z")
        a = self.B(ts=ts, cf=CustomFloat(1.))
        result = a.json()
        expected = '{"ts": "2023-04-12T00:00:00Z", "cf": 1.5}'
        self.assertEqual(expected, result)

    def test_custom_json_decoding(self):
        d = json.loads('{"ts": "2023-04-12T00:00:00Z", "cf": 1.5}')
        b = self.B(**d)
        self.assertIsInstance(b.ts, TS)
        self.assertIsInstance(b.cf, CustomFloat)
        self.assertEqual(b.cf, CustomFloat(1.5))
        self.assertEqual(b.ts, TS("2023-04-12T00:00:00Z"))

    def test_forbid_extra(self):
        with self.assertRaises(TypeError):
            self.A(ts=TS("2023-04-12T00:00:00Z"), extra_field="extra")

    def test_allow_arbitrary_type(self):
        class ArbitraryType:
            pass

        @dataclass
        class C(JsonData):
            at: ArbitraryType

        at = ArbitraryType()
        c = C(at=at)
        self.assertIsInstance(c.at, ArbitraryType)

    def test_mutability(self):
        @dataclass
        class C(JsonData):
            i: int

        c = C(i=1)
        c.i = 2
        self.assertEqual(c.i, 2)

    def test_pd_dataframe_member(self):
        @dataclass
        class C(JsonData):
            df: pd.DataFrame
            optional_df: Optional[pd.DataFrame] = None

        df = pd.DataFrame([1, 2, 3])
        c = C(df=df)
        self.assertIsInstance(c.df, pd.DataFrame)
        self.assertTrue(c.df.equals(df))
        self.assertIsNone(c.optional_df)

    def test_explicit_coercion(self):
        @dataclass
        class C(JsonData):
            i: int = coercing_field(default=0)

        c = C(i="1")
        self.assertEqual(c.i, 1)
        c = C()
        self.assertEqual(c.i, 0)

    def test_explicit_coercion_list(self):
        @dataclass
        class C(JsonData):
            i: int = coercing_field(default=0)

        @dataclass
        class D(JsonData):
            ds: List[C]
            tss: Optional[List[TS]]

        c = C(i="1")
        self.assertEqual(c.i, 1)
        ts = TS("2023-04-12T00:00:00Z")
        d = D(ds=[c, c], tss=[ts])
        j = d.json()
        d = json.loads(j)
        new_d = D(**d)
        # ToDo: implement handling of complex data types for coercion
        # self.assertIsInstance(new_d.ds, list)


class TestFrozenJsonData(TestCase):
    @dataclass(frozen=True)
    class A(JsonData):
        ts: TS = coercing_field()

        class Config:
            json_encoders = {
                TS: TS.as_iso.fget
            }

    @dataclass(frozen=True)
    class B(A):
        cf: CustomFloat = coercing_field()

        class Config:
            json_encoders = {
                CustomFloat: lambda cf: cf + 0.5, }

    def test_mutability(self):
        @dataclass(frozen=True)
        class C(JsonData):
            i: int

        c = C(i=1)
        with self.assertRaises(FrozenInstanceError):
            c.i = 2
