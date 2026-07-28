# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

version_number = "1.66.02"

# for PDF output on Read the Docs
project = "TransferBench Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"

extensions = ["rocm_docs"]
html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "rocm-extras",
    "header_title": f"ROCm™ TransferBench {version_number}",
    "header_link": f"https://rocm.docs.amd.com/projects/TransferBench/en/docs-1.66.02/",
    "link_main_doc": True,
}

external_projects_current_project = "transferbench"
