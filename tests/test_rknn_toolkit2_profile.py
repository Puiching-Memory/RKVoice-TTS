from __future__ import annotations

from scripts.testing.rknn_toolkit2_profile import extract_total_operator_elapsed_time_us, normalize_sdk_version
from tests.test_support import WorkspaceTestCase


class RKNNToolkit2ProfileTests(WorkspaceTestCase):
    def test_extract_total_operator_elapsed_time_us(self) -> None:
        text = """
        Performance
        Total Operator Elapsed Time(us): 14147
        Total Memory RW Amount(MB): 0
        """

        self.assertEqual(extract_total_operator_elapsed_time_us(text), 14147.0)

    def test_extract_total_operator_elapsed_time_us_supports_per_frame_summary(self) -> None:
        text = """
        Total Operator Elapsed Per Frame Time(us): 785166
        Total Memory Read/Write Per Frame Size(KB): 1001510.19
        """

        self.assertEqual(extract_total_operator_elapsed_time_us(text), 785166.0)

    def test_normalize_sdk_version_prefers_attributes(self) -> None:
        class Version:
            api_version = "1.0"
            drv_version = "2.0"

        self.assertEqual(normalize_sdk_version(Version()), {"api_version": "1.0", "drv_version": "2.0"})

    def test_normalize_sdk_version_falls_back_to_raw(self) -> None:
        self.assertEqual(normalize_sdk_version("demo-version"), {"raw": "demo-version"})