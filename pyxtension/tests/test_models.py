# Author: ASU --<andrei.suiu@gmail.com>
import json
from dataclasses import dataclass, FrozenInstanceError
from typing import Optional
from unittest import TestCase

import pandas as pd
from tsx import TS

from pyxtension.models import ExtModel, FrozenJsonData, JsonData, coercing_field, jsoned_data


class TestExtModel(TestCase):
    def test_json(self):
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

            class Config:
                json_encoders = {
                    CustomFloat: lambda cf: cf + 0.5, }

        ts = TS("2023-04-12T00:00:00Z")
        a = B(ts=ts, cf=CustomFloat(1.))
        result = a.json()
        expected = '{"ts": "2023-04-12T00:00:00Z", "cf": 1.5}'
        self.assertEqual(result, expected)


class CustomFloat(float):
    pass


class TestSmartDataclass(TestCase):
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


class TestFrozenSmartDataclass(TestCase):
    @dataclass(frozen=True)
    class A(FrozenJsonData):
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

    def test_mutability(self):
        @dataclass(frozen=True)
        class C(FrozenJsonData):
            i: int

        c = C(i=1)
        with self.assertRaises(FrozenInstanceError):
            c.i = 2


class TestJsonedData(TestCase):
    @jsoned_data(frozen=True, encoders={TS: TS.as_iso.fget})
    class A:
        ts: TS = coercing_field()

    @jsoned_data(frozen=True, encoders={CustomFloat: lambda cf: cf + 0.5, })
    class B(A):
        cf: CustomFloat = coercing_field()

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

    def test_mutability(self):
        @jsoned_data(frozen=True)
        class C:
            i: int

        c = C(i=1)
        with self.assertRaises(FrozenInstanceError):
            c.i = 2
