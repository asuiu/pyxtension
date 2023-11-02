# Author: ASU --<andrei.suiu@gmail.com>
import json
from dataclasses import dataclass, FrozenInstanceError
from unittest import TestCase

from tsx import TS

from pyxtension.models import ExtModel, FrozenSmartDataclass, SmartDataclass


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
    class A(SmartDataclass):
        ts: TS

        class Config:
            json_encoders = {
                TS: TS.as_iso.fget
            }

    @dataclass
    class B(A):
        cf: CustomFloat

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
        class C(SmartDataclass):
            at: ArbitraryType

        at = ArbitraryType()
        c = C(at=at)
        self.assertIsInstance(c.at, ArbitraryType)

    def test_mutability(self):
        @dataclass
        class C(SmartDataclass):
            i: int

        c = C(i=1)
        c.i = 2
        self.assertEqual(c.i, 2)


class TestFrozenSmartDataclass(TestCase):
    @dataclass(frozen=True)
    class A(FrozenSmartDataclass):
        ts: TS

        class Config:
            json_encoders = {
                TS: TS.as_iso.fget
            }

    @dataclass(frozen=True)
    class B(A):
        cf: CustomFloat

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
        class C(FrozenSmartDataclass):
            i: int

        c = C(i=1)
        with self.assertRaises(FrozenInstanceError):
            c.i = 2
