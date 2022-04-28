"""Report generators and templates."""

from lightautoml.utils.installation import __validate_extra_deps

from .report_deco_gpu import ReportDeco



__validate_extra_deps("pdf")


__all__ = ["ReportDeco"]
