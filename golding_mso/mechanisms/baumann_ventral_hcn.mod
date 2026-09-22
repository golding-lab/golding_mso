
NEURON {
	SUFFIX baumann_ventral_hcn USEION h READ eh WRITE ih
	RANGE gbar 
	RANGE ih,g
	
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gbar	= 0.017	(mho/cm2)  
	eh=-106		(mV) 
	celsius =22		(degC)
	dt              (ms)
	v               (mV)
	
}


ASSIGNED {
	g(mho/cm2)
	ih(mA/cm2)
}

STATE{
	m
}


BREAKPOINT {
	SOLVE states METHOD cnexp
	g = gbar * m
	ih = g*(v-eh)
}


INITIAL {

	m = m_inf(v)

	}

DERIVATIVE states {	
	m' = (m_inf(v)-m)/m_tau(v)
}

FUNCTION m_inf(Vm (mV)){
	UNITSOFF
	m_inf = 1/(1+exp(0.095*(Vm+75.5)))
	UNITSON
}


FUNCTION m_tau(Vm (mV)) (ms){
	UNITSOFF
	m_tau = 65+292*exp((-(Vm+62.5)^2)/722)
	UNITSON
}








