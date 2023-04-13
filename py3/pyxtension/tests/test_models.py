# Author: ASU --<andrei.suiu@gmail.com>
from unittest import TestCase

from tsx import TS

from py3.pyxtension.models import ExtModel


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
