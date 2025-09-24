import itertools
from typing import NamedTuple, Tuple, List
import argparse
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from nupack import Model, Strand, Complex, Tube, SetSpec, tube_analysis
import math

import nuad.constraints as nc  # type: ignore
import nuad.vienna_nupack as nv  # type: ignore
import nuad.search as ns  # type: ignore

from scipy.integrate import odeint

from strands import Signal, Drain, Template
from rates import compute_rates, polK, polV, polK_both, polV_both, bsmIV, bsmIK, nbIV, nbIK, exoV, exoK, stack


class System:
    """System class to manage the signals, templates and drains"""
    def __init__(self, 
                 temperature:float=39,
                 sodium:float=0.07,
                 magnesium:float=0.0125,
                 signals:List[Signal]=[], 
                 templates:List[Template]=[], 
                 drains:List[Drain]=[],
                 dict_input={},
                 dict_output={},
                 dict_drain={},
                 dict_kin_rates={},
                 ):
        self.temperature=temperature
        self.sodium=sodium
        self.magnesium=magnesium
        self.signals=signals
        self.templates=templates
        self.drains=drains
        self.dict_input=dict_input
        self.dict_output=dict_output
        self.dict_drain=dict_drain
        self.dict_kin_rates=dict_kin_rates

    def unmalloc(self):
        """Clean up the System instance by clearing references and mutable structures."""
        self.signals.clear()
        self.templates.clear()
        self.drains.clear()
        
        self.dict_input.clear()
        self.dict_output.clear()
        self.dict_drain.clear()
        self.dict_kin_rates.clear()
        
        self.signals = None
        self.templates = None
        self.drains = None
        self.dict_input = None
        self.dict_output = None
        self.dict_drain = None
        self.dict_kin_rates = None

        self.temperature = None
        self.sodium = None
        self.magnesium = None


    def add_signal(self,s:Signal):
        """Add a signal to the system"""
        if s not in self.signals:
            self.signals.append(s)

    def add_template(self,tmp:Template):
        """Add a template to the system"""
        if tmp not in self.templates:
            self.templates.append(tmp)
            if isinstance(tmp.input,Signal) and tmp.input not in self.signals:
                self.signals.append(tmp.input)
            elif isinstance(tmp.input,Drain) and tmp.input not in self.drains:
                self.drains.append(tmp.input)
            if isinstance(tmp.output,Signal) and tmp.output not in self.signals:
                self.signals.append(tmp.output)
            elif isinstance(tmp.output,Drain) and tmp.output not in self.drains:
                self.drains.append(tmp.output)

    def add_drain(self,drain:Drain):
        """Add a drain to the system"""
        if drain not in self.drains:
            self.drains.append(drain)
            if drain.input not in self.signals:
                self.signals.append(drain.input)

    def find_index(self,name,option="alone"):
        """Find the index of a signal, drain or template depending on its name"""
        for s in self.signals:
            if s.name == name:
                return s.get_index(option)
        for d in self.drains:
            if d.name==name:
                return d.get_index(option)
        for tmp in self.templates:
            if tmp.name==name:
                return tmp.get_index(option)
    
    def associate_index(self):
        """Associate an index to each signal, drain and non protected template"""
        i=0
        for s in self.signals:
            s.index = i
            if s.IsDrained:
                i+=1
            i+=1
        for  d in self.drains:
            d.index = i
            i+=3
        for tmp in self.templates:
            tmp.index = i
            i+=5
            if tmp.loading:
                i+=1
            if tmp.input.IsDrained:
                i+=1
        
    def nb_equations(self):
        """Return the number of equations"""
        nb_equations=len(self.signals) + len([s for s in self.signals if s.IsDrained]) + 3*len(self.drains) + 5*len(self.templates) 
        nb_equations+= len([t for t in self.templates if t.loading]) + len([t for t in self.templates if t.input.IsDrained])
        return nb_equations

    def update_dicts(self):
        """Update the reaction dictionnaries and the kinetic dictionnaries (containing association and dissociation rates k_on and k_off)"""
        self.dict_input.clear()
        self.dict_output.clear()
        self.dict_drain.clear()
        self.dict_kin_rates.clear()
        for s in self.signals:
            for tmp in self.templates:
                if tmp.input==s and tmp.output==s:
                    self.dict_kin_rates[(s,tmp,'input')]=compute_rates(s,tmp,self.temperature, self.sodium, self.magnesium, option='input')
                    self.dict_kin_rates[(s,tmp,'output')]=compute_rates(s,tmp,self.temperature, self.sodium, self.magnesium, option='output')
                if tmp.output==s:
                    self.dict_kin_rates[(s,tmp)]=compute_rates(s,tmp,self.temperature, self.sodium, self.magnesium)
                    if s in self.dict_output and tmp not in self.dict_output[s]:
                        self.dict_output[s].append(tmp)
                    else:
                        self.dict_output[s]=[tmp]
                if tmp.input==s:
                    self.dict_kin_rates[(s,tmp)]=compute_rates(s,tmp,self.temperature, self.sodium, self.magnesium)
                    if s in self.dict_input and tmp not in self.dict_input[s]:
                        self.dict_input[s].append((tmp,0))
                    else:
                        self.dict_input[s]=[(tmp,0)]
            for d in self.drains:
                if d.input==s:
                    self.dict_kin_rates[(s,d)]=compute_rates(s,d,self.temperature, self.sodium, self.magnesium)
                    self.dict_drain[s]=[d]
        for d in self.drains:
            self.dict_kin_rates[d]=compute_rates(d,None,self.temperature, self.sodium, self.magnesium)
            for tmp in self.templates:
                if tmp.output==d:
                    self.dict_kin_rates[(d,tmp)]=compute_rates(d,tmp,self.temperature, self.sodium, self.magnesium)
                    if d in self.dict_output and tmp not in self.dict_output[d]:
                        self.dict_output[d].append(tmp)
                    else:
                        self.dict_output[d]=[tmp]


    def init_equations(self):
        """Return the initial conditions of the equations"""
        nb_equations=self.nb_equations()
        y0=[0]*nb_equations
        for s in self.signals:
            y0[s.index]=s.concentration
        for d in self.drains:
            y0[d.index]=d.concentration
        for tmp in self.templates:
            y0[tmp.index]=tmp.concentration
        return y0


    
    def generate_equations(self,y,t):
        """Generate the equations, depending on the time step"""
        nb_equations=self.nb_equations()
        equations=[0]*nb_equations

        def k_on(s1:Signal|Drain|Template, s2:Signal|Drain|Template|None, option:str="default"):
            """Return the binding rate between the signal dn its complementary"""
            if s2==None:
                return self.dict_kin_rates[d][0]*1e-9
            if isinstance(s2,Template) and s2.input==s2.output:
                if option=='input':
                    return self.dict_kin_rates[(s1,s2,'input')][0]*1e-9
                else:
                    return self.dict_kin_rates[(s1,s2,'output')][0]*1e-9
            return self.dict_kin_rates[(s1,s2)][0]*1e-9
            # return 3.3e-3

        def k_off(s1:Signal|Drain|Template, s2:Signal|Drain|Template|None, option:str="default"):
            """Return the unbinding rate of a strand and its complementary"""
            if s2==None:
                return self.dict_kin_rates[d][1]
            if isinstance(s2,Template) and s2.input==s2.output:
                if option=='input':
                    return self.dict_kin_rates[(s1,s2,'input')][1]
                else:
                    return self.dict_kin_rates[(s1,s2,'output')][1]
            return self.dict_kin_rates[(s1,s2)][1]
            # return 0.35

        #Enzyme activity
        def pol(temp:Template|Drain):
            """Return the acrivity of the polymerase"""
            pol=polV(temp)
            div=1 + sum([tmp.get_concentration(y,"in")/polK(tmp) for tmp in self.templates]) # Use of pol for "alone" templates
            div+= sum([d.get_concentration(y,"in")/polK(d) for d in self.drains]) # Use of pol for drains
            div+= sum([tmp.get_concentration(y,"both")/polK_both(tmp) for tmp in self.templates]) # Use of pol for "both" templates 
            pol/=polK(temp)*div
            return pol

        def pol_both(temp:Template):
            """Return the activity of the polymerase when the output is already bound"""
            pol_both=polV_both(temp)
            div=1 + sum([tmp.get_concentration(y,"in")/polK(tmp) for tmp in self.templates]) # Use of pol for "alone" templates
            div+= sum([d.get_concentration(y,"in")/polK(d) for d in self.drains]) # Use of pol for drains
            div+= sum([tmp.get_concentration(y,"both")/polK_both(tmp) for tmp in self.templates]) # Use of pol for "both" templates
            pol_both/=polK_both(temp)*div
            return pol_both
        
        def bsmI(temp:Template):
            """Return the activity of the nicking enzyme bsmI with respect to the template"""
            enz=bsmIV(temp)
            div=1 + sum([tmp.get_concentration(y,"ext")/bsmIK(tmp) for tmp in self.templates if tmp.nick=="bsmI"])
            enz/=bsmIK(temp) * div
            return enz
        
        def nbI(temp:Template):
            """Return the activity of the nicking enzyme nbI"""
            enz=nbIV(temp)
            div=1 + sum([tmp.get_concentration(y,"ext")/nbIK(tmp) for tmp in self.templates if tmp.nick=="nbI"])
            enz/=nbIK(temp) * div
            return enz
        
        def exo(sig:Signal|Drain):
            """Return the activity of the exonuclease"""
            enz=exoV(sig)
            div= 1 + sum([s.get_concentration(y,"alone")/exoK(s) for s in self.signals])
            div+= sum([d.input.get_concentration(y,"drained")/exoK(d) for d in self.drains])
            div+= sum([d.get_concentration(y,"alone")/exoK(d) for d in self.drains])
            enz/=exoK(sig) * div
            return enz

        def flux_in(s:Signal,y):
            """Return the flux in for the signal s"""
            fluxin=0
            if s in self.dict_input:
                for tmp,_ in self.dict_input[s]:
                    if not tmp.loading:
                        fluxin+= k_off(s,tmp,'input') * tmp.get_concentration(y,"in") # The input unbinds from in
                        fluxin+= k_off(s,tmp,'input') * stack * tmp.get_concentration(y,"both") # The input unbinds from both
                        fluxin-= k_on(s,tmp,'input') * s.get_concentration(y,"alone") * tmp.get_concentration(y,"alone") # The input binds to alone
                        fluxin-= k_on(s,tmp,'input') * s.get_concentration(y,"alone") * tmp.get_concentration(y,"out") # The input binds to out
                    else:
                        fluxin+= k_off(s,tmp,'input') * tmp.get_concentration(y,"load") # The input unbinds from load
                        fluxin-= k_on(s,tmp,'input') * s.get_concentration(y,"alone") * tmp.get_concentration(y,"alone") # The input binds to alone


            if s.IsDrained and s in self.dict_drain:
                for d in self.dict_drain[s]:
                    fluxin+= k_off(s,d) * d.get_concentration(y,"in") # The input unbinds from in
                    fluxin-= k_on(s,d) * s.get_concentration(y,"alone") * (d.get_concentration(y,"alone")) #The input binds to alone
            return fluxin
        
        def flux_out(s:Signal|Drain,y):
            """Return the flux out for the signal s"""
            fluxout=0
            if s in self.dict_output:
                for tmp in self.dict_output[s]:
                    fluxout+= k_off(s,tmp,'output') * tmp.get_concentration(y,"out") # The output unbinds from out
                    fluxout+= k_off(s,tmp, 'output') * stack * tmp.get_concentration(y,"both") # The output unbinds from both
                    fluxout-= k_on(s,tmp, 'output') * s.get_concentration(y,"alone") * tmp.get_concentration(y,"alone") # The output binds to alone
                    fluxout-= k_on(s,tmp, 'output') * s.get_concentration(y,"alone") * tmp.get_concentration(y,"in") # The output binds to in
                    fluxout+= pol_both(tmp) * tmp.get_concentration(y,"both") # The output is displaced by the polymerase
            return fluxout


        for s in self.signals:
            gen=0
            gen+= flux_in(s,y)
            gen+= flux_out(s,y)
            if not s.protected:
                gen-= exo(s)*s.get_concentration(y,"alone") # The exonuclease binds to the signal
            equations[s.get_index("alone")]=gen
            #case IsDrained
            if s.IsDrained:
                gen=0
                if s in self.dict_drain:
                    for d in self.dict_drain[s]:
                        gen+= k_off(d,None) * d.get_concentration(y,"ext") # The output unbinds from drain Template
                        gen-= k_on(d,None) * d.get_concentration(y,"alone") * s.get_concentration(y,"drained") # The drained rebinds to the alone drain Template
                if s in self.dict_input:
                    for tmp,_ in self.dict_input[s]:
                        gen+= k_off(tmp.input, tmp, 'input')  * tmp.get_concentration(y,"in_drained") # The drained input unbinds from in_drained
                        gen-= k_on(tmp.input, tmp, 'input') * tmp.input.get_concentration(y,"drained") * tmp.get_concentration(y,"alone") # The drained input binds to alone
                gen-= exo(d)*d.input.get_concentration(y,"drained") # The exonuclease binds to drained
                equations[s.get_index("drained")]=gen

                

        for tmp in self.templates:
            if tmp.nick=="bsmI":
                nick=bsmI(tmp)
            elif tmp.nick=="nbI":
                nick=nbI(tmp)

            if not tmp.loading:
                # case alone
                gen = k_off(tmp.input,tmp,'input')  * tmp.get_concentration(y,"in") # The input unbinds from in
                gen+= k_off(tmp.output,tmp,'output')  * tmp.get_concentration(y,"out") # The output unbinds from out
                gen-= k_on(tmp.input,tmp,'input') * tmp.input.get_concentration(y,"alone") * tmp.get_concentration(y,"alone") # The input binds to alone
                gen-= k_on(tmp.output,tmp) * tmp.output.get_concentration(y,"alone") * tmp.get_concentration(y,"alone") # The output binds to alone
                if not tmp.protected:
                    gen-= exo(tmp.input)*tmp.get_concentration(y,"alone") # The exonuclease binds to alone
                if tmp.input.IsDrained:
                    gen+= k_off(tmp.input,tmp, 'input')  * tmp.get_concentration(y,"in_drained") # The drained input unbinds from in_drained
                    gen-= k_on(tmp.input, tmp, 'input') * tmp.input.get_concentration(y,"drained") * tmp.get_concentration(y,"alone") # The drained input binds to alone
                equations[tmp.get_index("alone")]=gen
                #case in
                gen = k_on(tmp.input,tmp,'input') * tmp.input.get_concentration(y,"alone") * tmp.get_concentration(y,"alone") # The input binds to alone
                gen+= k_off(tmp.output,tmp,'output') * stack * tmp.get_concentration(y,"both") # The output unbinds from both
                gen-= k_on(tmp.output,tmp,'output') * tmp.get_concentration(y,"in") * tmp.output.get_concentration(y,"alone") # The output binds to in
                gen-= k_off(tmp.input,tmp,'input') * tmp.get_concentration(y,"in")  # The input unbinds from in
                gen-= pol(tmp)*tmp.get_concentration(y,"in") # The polymerase binds to in
                equations[tmp.get_index("in")]=gen
                #case out
                gen = k_on(tmp.output,tmp,'output') * tmp.output.get_concentration(y,"alone") * tmp.get_concentration(y,"alone") # The output binds to alone
                gen+= k_off(tmp.input,tmp,'input')  * stack * tmp.get_concentration(y,"both") # The input unbinds from both
                gen-= k_on(tmp.input,tmp,'input') * tmp.get_concentration(y,"out") * tmp.input.get_concentration(y,"alone") # The input binds to out
                gen-= k_off(tmp.output,tmp,'output') * tmp.get_concentration(y,"out") # The output unbinds from out
                equations[tmp.get_index("out")]=gen
                #case both
                gen = k_on(tmp.input,tmp,'input') * tmp.input.get_concentration(y,"alone") * tmp.get_concentration(y,"out") # The input binds to out
                gen+= k_on(tmp.output,tmp,'output') * tmp.output.get_concentration(y,"alone") * tmp.get_concentration(y,"in") # The output binds to in
                gen-= k_off(tmp.input,tmp,'input') * stack * tmp.get_concentration(y,"both")  # The input unbinds from both
                gen-= k_off(tmp.output,tmp,'output') * stack * tmp.get_concentration(y,"both") # The output unbinds from both
                gen+= nick*tmp.get_concentration(y,"ext") # The nicking enzyme binds to ext
                gen-= pol_both(tmp)*tmp.get_concentration(y,"both") # The polymerase binds to both
                equations[tmp.get_index("both")]=gen
                #case ext
                gen = pol(tmp) * tmp.get_concentration(y,"in") # The polymerase binds to in
                gen+= pol_both(tmp) * tmp.get_concentration(y,"both") # The polymerase binds to both
                gen-= nick*tmp.get_concentration(y,"ext") # The nicking enzyme binds to ext
                equations[tmp.get_index("ext")]=gen
                #case in_drained
                if tmp.input.IsDrained:
                    gen = k_on(tmp.input, tmp,'input') * tmp.input.get_concentration(y,"drained") * tmp.get_concentration(y,"alone") # The drained input binds to alone
                    gen-= k_off(tmp.input,tmp,'input')  * tmp.get_concentration(y,"in_drained") # The drained input unbinds from in_drained
                    equations[tmp.get_index("in_drained")]=gen
                    

            else:
                # case alone
                gen = k_off(tmp.input,tmp,'input')  * tmp.get_concentration(y,"load") # The input unbinds from load
                gen+= k_off(tmp.output,tmp,'output')  * tmp.get_concentration(y,"out") # The output unbinds from out
                gen-= k_on(tmp.input,tmp,'input') * tmp.input.get_concentration(y,"alone") * tmp.get_concentration(y,"alone") # The input binds to alone
                gen-= k_on(tmp.output,tmp,'output') * tmp.output.get_concentration(y,"alone") * tmp.get_concentration(y,"alone") # The output binds to alone
                if not tmp.protected:
                    gen-= exo(tmp.input)*tmp.get_concentration(y,"alone") # The exonuclease binds to alone
                if tmp.input.IsDrained:
                    gen+= k_off(tmp.input,tmp,'input')  * tmp.get_concentration(y,"in_drained") # The drained input unbinds from in_drained
                    gen-= k_on(tmp.input, tmp,'input') * tmp.input.get_concentration(y,"drained") * tmp.get_concentration(y,"alone") # The drained input binds to alone
                equations[tmp.get_index("alone")]=gen
                #case load
                gen = k_on(tmp.input,tmp,'input') * tmp.input.get_concentration(y,"alone") * tmp.get_concentration(y,"alone") # The input binds to alone
                gen-= k_off(tmp.input,tmp,'input')  * tmp.get_concentration(y,"load") # The input unbinds from load
                gen-= pol(tmp)*tmp.get_concentration(y,"load") # The polymerase binds to load
                equations[tmp.get_index("load")]=gen
                #case in
                gen = k_off(tmp.output,tmp,'output') * stack * tmp.get_concentration(y,"both") # The output unbinds from both
                gen-= k_on(tmp.output,tmp,'output') * tmp.get_concentration(y,"in") * tmp.output.get_concentration(y,"alone") # The output binds to in
                gen-= pol(tmp)*tmp.get_concentration(y,"in") # The polymerase binds to in
                equations[tmp.get_index("in")]=gen
                #case out
                gen = k_on(tmp.output,tmp,'output') * tmp.output.get_concentration(y,"alone") * tmp.get_concentration(y,"alone") # The output binds to alone
                gen-= k_off(tmp.output,tmp,'output') * tmp.get_concentration(y,"out")  # The output unbinds from out
                equations[tmp.get_index("out")]=gen
                #case both
                gen = k_on(tmp.output,tmp,'output') * tmp.output.get_concentration(y,"alone") * tmp.get_concentration(y,"in") # The output binds to in
                gen-= k_off(tmp.output,tmp,'output') * stack * tmp.get_concentration(y,"both") # The output unbinds from both
                gen+= nick*tmp.get_concentration(y,"ext") # The nicking enzyme binds to ext
                gen-= pol_both(tmp) * tmp.get_concentration(y,"both") # The polymerase binds to both
                equations[tmp.get_index("both")]=gen
                #case ext
                gen = pol(tmp) * tmp.get_concentration(y,"in") # The polymerase binds to in
                gen+= pol(tmp) * tmp.get_concentration(y,"load") # The polymerase binds to load
                gen+= pol_both(tmp) * tmp.get_concentration(y,"both") # The polymerase binds to both
                gen-= nick*tmp.get_concentration(y,"ext") # The nicking enzyme binds to ext
                equations[tmp.get_index("ext")]=gen
                #case in_drained
                if tmp.input.IsDrained:
                    gen = k_on(tmp.input, tmp, 'input') * tmp.input.get_concentration(y,"drained") * tmp.get_concentration(y,"alone") # The drained input binds to alone
                    gen-= k_off(tmp.input,tmp, 'input')  * tmp.get_concentration(y,"in_drained") # The drained input unbinds from in_drained
                    equations[tmp.get_index("in_drained")]=gen
                    

        #drains
        for d in self.drains:
            #case alone
            gen = k_off(d,None) * d.get_concentration(y,"ext") # The drained signal unbinds from the pseudoTemplate
            gen+= k_off(d.input,d)  * d.get_concentration(y,"in") # The input unbinds from in
            gen-= k_on(d.input,d) * d.input.get_concentration(y,"alone") * d.get_concentration(y,"alone") # The input binds to alone
            gen-= k_on(d,None) * d.get_concentration(y,"alone") * d.input.get_concentration(y,"drained") # The drained signal rebinds to alone
            gen+=flux_out(d,y) # The created drains
            if not d.protected:
                gen-=exo(d)*d.get_concentration(y,"alone") # The exonuclease binds to alone
            equations[d.get_index("alone")]=gen
            #case in
            gen = k_on(d.input,d) * d.input.get_concentration(y,"alone") * d.get_concentration(y,"alone") # The input binds to alone
            gen-= k_off(d.input,d) * d.get_concentration(y,"in") # The input unbinds from in
            gen-= pol(d) * d.get_concentration(y,"in") # The polymerase binds to in
            equations[d.get_index("in")]=gen
            #case ext
            gen = k_on(d,None) * d.get_concentration(y,"alone") * d.input.get_concentration(y,"drained") # The drained rebinds to alone
            gen-= k_off(d,None) * d.get_concentration(y,"ext") # The drained signal unbinds from the pT
            gen+= pol(d) * d.get_concentration(y,"in") # The polymerase binds to in
            equations[d.get_index("ext")]=gen

        return equations

    def solve_system(self, t:List[float]):
        """Solve the system of equations"""
        self.associate_index()
        self.update_dicts()
        y0=self.init_equations()
        fun=lambda y,t : self.generate_equations(y,t)
        y = odeint(fun, y0, t)
        return y
    
    def solve_and_plot(self, time:List[float], species:List[Tuple[str,str]], show=True, save=False, path="results.png", args=None):
        """Plot the results"""
        y=self.solve_system(time)
        plt.figure(figsize=(10, 6))
        for s in species:
            if s[1]=='alone':
                label=s[0]
            else:
                label=s[0] + "_" + s[1]
            plt.plot(time, y[:, self.find_index(s[0],s[1])], label=label)
        plt.xlabel('Time (s)')
        plt.ylabel('Concentration (nM)')
        plt.legend()
        plt.title('Concentration vs Time')
        if save:
            plt.savefig(path)
        if show:
            plt.show()
        