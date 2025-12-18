NEURON { 
	SUFFIX leak_klt
	USEION k READ ek WRITE ik
	RANGE ik, gbar
}

PARAMETER {
	gbar = 0.001 (siemens/cm2) < 0, 1e9 >
	ek (millivolt)
}

ASSIGNED {
	ik (milliamp/cm2)
	v (millivolt)
}

BREAKPOINT { ik = gbar*(v-ek) } 

