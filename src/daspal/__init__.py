# SPDX-FileCopyrightText: 
# 2025-present Florian Le Pape <florian.le.pape@ifremer.fr>
# 2025-present Gwenaël CAËR <gwenael.caer@data-terra.org>
# 
# SPDX-License-Identifier: MIT


from . import _accessors

from .core import (
    find_files,
    load_data,
    merge,
    partition,
    save
)

#import logging

#logging.getLogger(__name__).addHandler(logging.NullHandler())