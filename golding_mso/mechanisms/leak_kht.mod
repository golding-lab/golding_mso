NEURON { 
	SUFFIX leak_kht
	USEION k READ ek WRITE ik
	RANGE ik, gbar
}

PARAMETER {
	gbar = 0.001 (siemens/cm2) < 0, 1e9 >
	ek = -65 (millivolt)
}

ASSIGNED {
	ik (milliamp/cm2)
	v (millivolt)
}

BREAKPOINT { ik = gbar*(v-ek) } 

