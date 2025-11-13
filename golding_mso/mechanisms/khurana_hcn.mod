

NEURON {
SUFFIX khurana_hcn	USEION h  READ eh  WRITE ih VALENCE 1  RANGE gbar, ih,gh }





UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}



PARAMETER {
gbar= 0.005	(mho/cm2) eh= -39 (mV)

} 


ASSIGNED {
gh(mho/cm2)
	ih(mA/cm2)
 v (mV)
}
    


STATE {
	rf rs }




BREAKPOINT {
SOLVE states
 METHOD cnexp 
gh=gbar*(0.65*rf +(1-0.65)*rs) 
ih=gh*(v-eh)
}






INITIAL {
	rf= r_inf(v)
	rs = r_inf(v)

}


DERIVATIVE states{
	rf'= (r_inf(v) - rf)/tau_f(v)
	rs'= (r_inf(v) - rs)/tau_s(v)   }

FUNCTION r_inf(Vm (mV)) () {
	UNITSOFF 
	r_inf = 1/(1+exp((Vm+60.3)/(7.3)))
	UNITSON
}


FUNCTION tau_f(Vm (mV)) (ms) {
	UNITSOFF 
	tau_f= 10000/( (-7.4*(Vm + 60)/(exp(-(Vm+60)/0.8)-1 ))   +   65*exp(-(Vm+56)/23))
	UNITSON
}

FUNCTION tau_s(Vm (mV)) (ms) {
	UNITSOFF 
	tau_s= 1000000/( (-56*(Vm + 59)/(exp(-(Vm+59)/0.8)-1 ))   +   0.24*exp(-(Vm-68)/16))
	UNITSON
}





