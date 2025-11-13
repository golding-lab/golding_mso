NEURON {
SUFFIX spencer_na
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
ena = 55 (mV)
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
g = gbar * m^3 * h
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
m_inf = 1/(1+-1.11*(Vm+58)*((Vm+49)^-1)*(exp(Vm+(49/-3))-1)*(exp(Vm+(58/20))-1)^-1)
UNITSON
}

FUNCTION h_inf(Vm (mV)) {
UNITSOFF
h_inf = (2.4*((1+exp((Vm+68)/3))^-1)+0.8*(1+exp(Vm+61.3))^-1)/(2.4*((1+exp((Vm+68)/3))^-1)+0.8*((1+exp(Vm+61.3))^-1)+3.6*((1+exp(-(Vm+21)/10))^-1))
UNITSON
}

FUNCTION m_tau(Vm (mV)) (ms) {
UNITSOFF
m_tau = 1/(-0.36*(Vm+49)*((exp(Vm+49/-4)-1)^-1)+0.4*(Vm+58)*((exp(Vm+58/20)-1)^-1))
UNITSON
}

FUNCTION h_tau(Vm (mV)) (ms) {
UNITSOFF
h_tau = 1/(2.4*((1+exp(Vm+(68/3)))^-1)+0.8*((1+exp(Vm+61.3))^-1)+3.6*(1+exp(-(Vm+21)/10))^-1)
UNITSON
}
