NEURON { 
	SUFFIX leak 
	NONSPECIFIC_CURRENT i 
	RANGE i, e, gbar
}

PARAMETER {
	gbar = 0.001 (siemens/cm2) < 0, 1e9 >
	e = -65 (millivolt)
}

ASSIGNED {
	i (milliamp/cm2)
	v (millivolt)
}

BREAKPOINT { i = gbar*(v-e) } 

