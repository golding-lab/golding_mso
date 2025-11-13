
NEURON {
	SUFFIX mathews_klt USEION k READ ek WRITE ik
	RANGE gbar 
	RANGE ik,g
	
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gbar	= 0.017	(mho/cm2)  
	ek=-106		(mV) 
	celsius =22		(degC)
	dt              (ms)
	v               (mV)
	
}


ASSIGNED {
	g(mho/cm2)
	ik(mA/cm2)
}

STATE{
	m
	h
}


BREAKPOINT {
	SOLVE states METHOD cnexp
	g = gbar * m^4 * h
	ik = g*(v-ek)
}


INITIAL {

	m = m_inf(v)
	h = h_inf(v)

	}

DERIVATIVE states {	
	m' = (m_inf(v)-m)/m_tau(v)
	h' = (h_inf(v)-h)/h_tau(v)
}

FUNCTION m_inf(Vm (mV)){
	UNITSOFF
	m_inf = 1/(1+exp(((Vm-(-57.34))/(-11.7))))
	UNITSON
}

FUNCTION h_inf(Vm (mV)){
	UNITSOFF
	h_inf = ((1-0.27)/(1+exp(((Vm-(-67))/(6.16)))))+0.27
	UNITSON
}

FUNCTION m_tau(Vm (mV)) (ms){
	UNITSOFF
	m_tau = (21.5/((6*exp((Vm+60)/7))+(24*exp(-(Vm+60)/50.6))))+0.35
	UNITSON
}

FUNCTION h_tau(Vm (mV)) (ms){
	UNITSOFF
	h_tau = (170/((5*exp((Vm+60)/10))+(exp(-(Vm+70)/8))))+10.7
	UNITSON
}







