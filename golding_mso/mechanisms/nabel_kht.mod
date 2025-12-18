NEURON {
SUFFIX nabel_kht
USEION k READ ek WRITE ik
RANGE gbar, g, ik
}

UNITS {
(S) = (siemens)
(mV) = (millivolt)
(mA) = (milliamp)
}

PARAMETER { 
gbar = 0.0005449 (S/cm2)
ek = -90 (mV)
x_tau = 0.8 (ms)
}

ASSIGNED {
v (mV)
ina (mA/cm2)
ik (mA/cm2)
g (S/cm2)
}

STATE { 
x
}

BREAKPOINT {


SOLVE states METHOD cnexp
g = gbar * x^2
ik = g * (v - ek)
}

INITIAL {
: Assume v has been constant for a long time
x = x_inf(v)
}

DERIVATIVE states {
: Computes state variable x at present v & t
x' = (x_inf(v)-x)/x_tau
}

FUNCTION x_inf(Vm (mV)){
UNITSOFF
x_inf = 1/(1+exp(-(Vm+44.9)/30))
UNITSON
}


