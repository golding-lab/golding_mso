NEURON {
SUFFIX scott_na
USEION na READ ena WRITE ina
RANGE gbar, g, i
}

UNITS {
(S) = (siemens)
(mV) = (millivolt)
(mA) = (milliamp)
}

PARAMETER { 
gbar = 0.036 (S/cm2)
ena = 59 (mV)
}

ASSIGNED {
v (mV)
ina (mA/cm2)
i (mA/cm2)
g (S/cm2)
}

STATE { 
m
h 
}

BREAKPOINT {


SOLVE states METHOD cnexp
g = gbar * m^4 * ((0.993 * h) + 0.007)
ina = g * (v - ena)
}

INITIAL {
: Assume v has been constant for a long time
m = m_inf(v)
h = h_inf(v)
}

DERIVATIVE states {
: Computes state variable m and h at present v & t
m' = (m_inf(v)-m)/m_tau(v)
h' = (h_inf(v)-h)/h_tau(v)
}

FUNCTION m_inf(Vm (mV)) {
UNITSOFF
m_inf = 1/(1+exp((Vm+46)/(-11)))
UNITSON
}

FUNCTION h_inf(Vm (mV)) {
UNITSOFF
h_inf = 1/(1+exp((Vm+62.5)/(7.77)))
UNITSON
}

FUNCTION m_tau(Vm (mV)) (ms) {
UNITSOFF
m_tau = (0.141 + (-0.0826/(1+exp((-20.5 - Vm)/10.8))))/3
UNITSON
}

FUNCTION h_tau(Vm (mV)) (ms) {
UNITSOFF
h_tau = (4 + (-3.74/(1+exp((-40.6 - Vm)/5.05))))/3
UNITSON
}
