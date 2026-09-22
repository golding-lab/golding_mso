NEURON { 
	SUFFIX leak_hcn
	USEION h READ eh WRITE ih
	RANGE ih, gbar
}

PARAMETER {
	gbar = 0.001 (siemens/cm2) < 0, 1e9 >
	eh (millivolt)
}

ASSIGNED {
	ih (milliamp/cm2)
	v (millivolt)
}

BREAKPOINT { ih = gbar*(v-eh) } 

