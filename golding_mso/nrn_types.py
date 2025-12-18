"""
Type creation for NEURON objects, used in type hinting.
"""

from neuron import nrn, hoc
from typing import NewType

Section = NewType("Section", nrn.Section)
Segment = NewType("Segment", nrn.Segment)
Exp2Syn = NewType("Exp2Syn", hoc.HocObject)
NetCon = NewType("NetCon", hoc.HocObject)
NetStim = NewType("NetStim", hoc.HocObject)
MechanismStandard = NewType("MechanismStandard", hoc.HocObject)
Vector = NewType("Vector", hoc.HocObject)
