import itertools
from typing import NamedTuple, Tuple, List
import argparse
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from nupack import Model, Strand, Complex, Tube, SetSpec, tube_analysis
import math
import networkx as nx


import nuad.constraints as nc  # type: ignore
import nuad.vienna_nupack as nv  # type: ignore
import nuad.search as ns  # type: ignore

from scipy.integrate import odeint

from strands import Signal, Drain, Template, Reporter, reverse_complement
from rates import compute_rates, polK, polV, polK_both, polV_both, BsmIV, BsmIK, NBIV, NBIK, exoV, exoK, stack_slowdown


class PEN_System:
    """PEN_System class to manage the signals, templates and drains"""
    def __init__(self,
             temperature: float = 39,
             sodium: float = 0.07,
             magnesium: float = 0.0125,
             concentration_BsmI: float|None = None,
             concentration_NBI: float|None = None,
             concentration_pol: float|None = None,
             concentration_exo: float|None = None,
             signals: List[Signal] = None,
             templates: List[Template] = None,
             drains: List[Drain] = None,
             reporters: List[Reporter] = None,
             nb_equations=0,
             dict_index=None,
             dict_input=None,
             dict_output=None,
             dict_drain=None,
             dict_kin_rates=None,
             dict_stack=None,
             dict_tmp_elongation=None,
             dict_var_signals=None,
             leak: float = 0,
             ):
        self.temperature = temperature
        self.sodium = sodium
        self.magnesium = magnesium
        self.concentration_BsmI = concentration_BsmI
        self.concentration_NBI = concentration_NBI
        self.concentration_pol = concentration_pol
        self.concentration_exo = concentration_exo
        self.signals = signals if signals is not None else []             # List of all the signals
        self.templates = templates if templates is not None else []       # List of all the templates
        self.drains = drains if drains is not None else []                # List of all the drains
        self.reporters = reporters if reporters is not None else []       # List of all the reporters
        self.dict_index = dict_index if dict_index is not None else {}    # Dictionary that associates its index to each oligos or duplex of oligos 
        self.dict_input = dict_input if dict_input is not None else {}    # Dict that associates all the inputting templates of a signal 
        self.dict_output = dict_output if dict_output is not None else {} # Same for outputting templates
        self.dict_drain = dict_drain if dict_drain is not None else {}    # Same for drain templates
        self.dict_tmp_elongation = dict_tmp_elongation if dict_tmp_elongation is not None else {}  # Dictionary that maps a template to its elongated version (if any)
        self.dict_var_signals = dict_var_signals if dict_var_signals is not None else {}  # Dictionary that maps a signal name to all the versions of this template
        self.dict_kin_rates = dict_kin_rates if dict_kin_rates is not None else {}  # Dictionary of all the hybridization rates 
        self.dict_stack = dict_stack if dict_stack is not None else {}  # Dictionary of all the coaxial stacking slowdown 
        self.leak = leak
        self.nb_equations=nb_equations


    def unmalloc(self):
        """Clean up the PEN_System instance by clearing references and mutable structures."""
        self.signals.clear()
        self.templates.clear()
        self.drains.clear()
        self.reporters.clear()
        
        self.dict_index.clear()
        self.dict_input.clear()
        self.dict_output.clear()
        self.dict_drain.clear()
        self.dict_kin_rates.clear()
        self.dict_tmp_elongation.clear()
        self.dict_var_signals.clear()
    
        
        self.signals = None
        self.templates = None
        self.drains = None
        self.dict_input = None
        self.dict_output = None
        self.dict_drain = None
        self.dict_kin_rates = None
        self.dict_tmp_elongation = None

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


    def add_reporter(self, rep:Reporter):
        """Add a reporter to the system"""
        if rep not in self.reporters:
            self.reporters.append(rep)
            if rep.input not in self.signals:
                self.signals.append(rep.input)
            if rep.output and rep.output not in self.signals:
                self.signals.append(rep.output)
            
    def print(syst, enzymes_kinetic=False, detailed_kinetics=False):
        """
        Print a detailed and well-formatted summary of a PEN_System instance.

        Args:
            syst (PEN_System): The PEN_System instance to print.
        """
        # Print scalar attributes
        print("=" * 50)
        print("PEN System Summary")
        print("=" * 50)
        print(f"Temperature: {syst.temperature} °C")
        print(f"Leak rate: {syst.leak}")
        print(f"Sodium concentration: {syst.sodium} M")
        print(f"Magnesium concentration: {syst.magnesium} M\n")

        if enzymes_kinetic:
            print("-" * 50)
            print("Enzyme Concentrations:")
            print(f"  Nb.BsmI concentration: {syst.concentration_BsmI} U/µL")
            print(f"  Nt.BstNBI concentration: {syst.concentration_NBI} U/µL")
            print(f"  Polymerase concentration: {syst.concentration_pol} U/µL")
            print(f"  Exonuclease concentration: {syst.concentration_exo} nM\n")
            print()

        # Print dict_kin_rates as a formatted table
        if detailed_kinetics:
            print("-" * 50)
            print("Kinetic Rates Dictionary")
            print("-" * 50)
            if syst.dict_kin_rates:
                for key, value in syst.dict_kin_rates.items():
                    # Handle different key types
                    if isinstance(key, tuple):
                        if len(key) == 2:
                            obj1, obj2 = key
                            # Get the names of the objects
                            name1 = getattr(obj1, 'name', str(obj1))
                            name2 = getattr(obj2, 'name', str(obj2)) if obj2 is not None else "None"
                            # Print the names and the rates
                            if isinstance(value, (tuple, list)):
                                print(f"{name1:<15} <-> {name2:<15} : k_on = {value[0]:.2e} nM⁻¹·min⁻¹, k_off = {value[1]:.2e} min⁻¹")
                            else:
                                print(f"{name1:<15} <-> {name2:<15} : rate = {value:.2e}")
                        elif len(key) == 3:
                            # Handle keys like (Signal, Template, 'input') or (Signal, Template, 'output')
                            obj1, obj2, option = key
                            if isinstance(obj1,Signal):
                                name1=getattr(obj1, 'name', str(obj1))
                                if obj1.plus!=0:
                                    name1+='+'+str(getattr(obj1, 'plus', str(obj1)))
                                if obj1.minus!=0:
                                    name1+='-'+str(getattr(obj1, 'minus', str(obj1)))
                                name2 = getattr(obj2, 'name', str(obj2))
                            if isinstance(value, (tuple, list)):
                                print(f"{name1:<15} <-> {name2:<15} ({option}) : k_on = {value[0]:.2e} nM⁻¹·min⁻¹, k_off = {value[1]:.2e} min⁻¹")
                            else:
                                print(f"{name1:<15} <-> {name2:<15} ({option}) : rate = {value:.2e}")
                    else:
                        # Handle keys that are not tuples (e.g., Drain objects)
                        name = getattr(key, 'name', str(key))
                        if hasattr(key, 'input') and hasattr(key.input, 'name'):
                            # This is a Drain object with an input signal
                            drained_signal_name = f"{key.input.name}_drained"
                            if isinstance(value, (tuple, list)):
                                print(f"{name:<15} <-> {drained_signal_name:<15} : k_on = {value[0]:.2e} nM⁻¹·min⁻¹, k_off = {value[1]:.2e} min⁻¹")
                            else:
                                print(f"{name:<15} <-> {drained_signal_name:<15} : rate = {value:.2e}")
                        else:
                            if isinstance(value, (tuple, list)):
                                print(f"{name:<15} : k_on = {value[0]:.2e} nM⁻¹·min⁻¹, k_off = {value[1]:.2e} min⁻¹")
                            else:
                                print(f"{name:<15} : rate = {value:.2e}")
                print("-" * 50)
                for key, value in syst.dict_stack.items():
                    name = getattr(key, 'name', str(key))
                    print(f"{name:<15} : rate = {value:.2e}")
            else:
                print("No kinetic rates defined.")
            print()


        # Print Signals
        if len(syst.signals)>0:
            print("-" * 50)
            print("List of Signals")
            print("-" * 50)
            if syst.signals:
                for i, s in enumerate(syst.signals, 1):
                    print(f"Signal {i}:")
                    s.print()
                    print()
            else:
                print("No signals defined.")
            print()

        # Print Drains
        if len(syst.drains)>0:
            print("-" * 50)
            print("List of Drains")
            print("-" * 50)
            if syst.drains:
                for i, drain in enumerate(syst.drains, 1):
                    print(f"Drain {i}:")
                    drain.print()
                    print()
            else:
                print("No drains defined.")
            print()

        # Print Templates
        if len(syst.templates)>0:
            print("-" * 50)
            print("List of Templates")
            print("-" * 50)
            if syst.templates:
                for i, tmp in enumerate(syst.templates, 1):
                    print(f"Template {i}:")
                    tmp.print()
                    print()
            else:
                print("No templates defined.")
            print()


        # Print Reporters
        if len(syst.reporters)>0:
            print("-" * 50)
            print("List of Reporters")
            print("-" * 50)
            if syst.reporters:
                for i, rep in enumerate(syst.reporters, 1):
                    print(f"Reporter {i}:")
                    rep.print()
                    print()
            else:
                print("No reporters defined.")
            print()


    def graph(self):
        """
        Experimental code
        Build the graph representation of the system, with signals, templates and drains as nodes, and edges between them if they interact in the system
        """
        G=nx.DiGraph()
        wastes=[rT.output for rT in self.reporters if rT.output!=None]
        for sig in self.signals:
            G.add_node(sig.name, type="Signal", sequence=sig.sequence, concentration=sig.concentration, waste=sig in wastes)
        for dr in self.drains:
            if dr in self.dict_output:
                pass
            else:
                G.add_node(dr.name, type="Drain", sequence=dr.sequence, concentration=dr.concentration)
                G.add_edge(dr.input.name, dr.name, type="Drain", name=dr.name)
        for tmp in self.templates:
            if not tmp.input or not tmp.output:
                raise Exception(f"{tmp.name} has no input or no output")
            if tmp.input.name not in G:
                G.add_node(tmp.input.name, type="Signal", sequence=tmp.input.sequence, concentration=tmp.input.concentration)
            if tmp.output.name not in G and not isinstance(tmp.output, Drain):
                G.add_node(tmp.output.name, type="Signal", sequence=tmp.output.sequence, concentration=tmp.output.concentration)
            if isinstance(tmp.output, Drain):
                G.add_edge(tmp.input.name, tmp.output.input.name, type="kT", name=tmp.name)
            elif isinstance(tmp.output, Signal):
                G.add_edge(tmp.input.name, tmp.output.name, type="Template", name=tmp.name)
        for rT in self.reporters:
            G.add_edge(rT.input.name, rT.output.name, type="rT", name=rT.name)



        return G
    
    def graph_representation(self, label:bool=False, size=1):
        """
        Experimental code
        Plot the graph representation of the system, with signals, templates and drains as nodes, and edges between them if they interact in the system. 
        The color and shape of the nodes depend on their type (signal, template or drain), and the color of the edges depend on the type of interaction (kT or template).
        """
        G=self.graph()
        # Choose a layout
        plt.figure(figsize=(20*size, 15*size))
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        node_sizes = []
        node_colors = []
        node_shapes = []
        for node in G.nodes():
            if G.nodes[node].get("type") == "Drain":
                node_sizes.append(100)  # Small size
                node_colors.append("black")
                node_shapes.append("o")
            elif G.nodes[node].get("waste"):
                node_sizes.append(300)  # Default size
                node_colors.append("yellow")  # Yellow color for waste signals
                node_shapes.append("*")  # Star shape for waste signals
            else:
                node_sizes.append(300)  # Default size
                node_colors.append("skyblue")  # Default color
                node_shapes.append("o")
        
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

        # Draw nodes
        for shape in set(node_shapes):
            nodes_of_shape = [i for i, s in enumerate(node_shapes) if s == shape]
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[list(G.nodes())[i] for i in nodes_of_shape],
                node_size=[node_sizes[i] for i in nodes_of_shape],
                node_color=[node_colors[i] for i in nodes_of_shape],
                node_shape=shape,
                                      )

        # Extract edge types (assuming edge types are stored as attributes)
        edge_colors = []
        arrow_styles = []
        for u, v, attr in G.edges(data=True):
            if attr.get("type") == "kT":
                edge_colors.append("red")
                arrow_styles.append("-[")
            else:
                edge_colors.append("black")  # or any other default color
                arrow_styles.append("-|>")  # or any other default style

        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            arrows=True,
            arrowstyle=arrow_styles,
        )

        # nx.draw(G, pos, with_labels=True, arrows=True, edge_color=edge_colors, node_size=node_sizes, node_color=node_colors, node_shape=node_shapes, font_size=10, font_weight="bold")
            # Draw node labels in bold
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight="bold",  # Bold labels
            labels={node: node for node in G.nodes() if not G.nodes[node].get("waste")},  # Only label signals
        )

        # Labels
        if label:
            edge_labels = { e: G.edges[e].get("name", "") for e in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

        plt.axis("off")
        plt.show()



    def IsWellDefined(self):
        """Depricated"""
        # 1: Assert all the signals are properly defined
        for s in self.signals:
            if not isinstance(s,Signal):
                raise Exception(f"{s} isn't properly defined")

        # 2: Assert all the templates have a defined input and output
        for tmp in self.templates:
            if not isinstance(tmp,Template):
                raise Exception(f"{tmp} is not a Template")
            if (not isinstance(tmp.input, Signal)) or tmp.input not in self.signals:
                raise Exception(f"Error, {tmp.input} isn't defined")
            if (not (isinstance(tmp.output, Signal) or isinstance(tmp.output, Drain))) or tmp.output not in self.signals:
                raise Exception(f"Error, {tmp.output} isn't defined")
        
        # 3: Assert all the Drains have a defined input
        for d in self.drains:
            if not isinstance(d,Drain):
                raise Exception(f"{d} isn't properly defined")
            if (not isinstance(d.input, Signal)) or d.input not in self.signals:
                raise Exception(f"Error, {tmp.input} isn't defined")


        # 4: Assert all the sequences are distinct

        # 5: Assert the sequences correctly match

        # TODO

        
        return True


    def add_missing_oligos(self):
        """
        Add missing oligos to the system, such as elongated templates for non-phosphorylated templates,
        and shorter versions of signals with both plus and minus modifications.
        """
        for tmp in self.templates:
            if tmp.sequence!=None and not tmp.phosphorylated and not tmp.irreversible:
                sig_in = tmp.input
                sig_out=tmp.output
                rev_seq= reverse_complement(tmp.sequence)
                seq_beg_tmp = rev_seq[:len(sig_out.sequence)]
                if seq_beg_tmp != sig_in.sequence:
                    elongated_seq=reverse_complement(sig_in.sequence + sig_out.sequence)
                    elongated_tmp= None
                    for tmp2 in self.templates:
                        if tmp2.sequence == elongated_seq:
                            elongated_tmp=tmp2
                            break
                    if elongated_tmp == None:
                        elongated_tmp = Template(name=f"{tmp.name}_elongated",
                                                 input= tmp.input,
                                                 output= tmp.output,
                                                 concentration=0,
                                                 sequence=elongated_seq,
                                                 protected=tmp.protected,
                                                 phosphorylated=False,
                                                 irreversible=False,
                                                 nick=tmp.nick)
                        self.add_template(elongated_tmp)
                    self.dict_tmp_elongation[tmp]=elongated_tmp
        for s in self.signals:
            if s.minus != 0 and s.plus!=0:
                sig_shorter = None
                for s2 in self.signals:
                    if s2.name==s.name and s2.plus==0 and s2.minus==s.minus:
                        sig_shorter=s2
                if sig_shorter==None:
                    sig_shorter = Signal(name=s.name,
                                            concentration=0,
                                            sequence=s.sequence[:-s.plus],
                                            protected=False,
                                            plus=0,
                                            minus=s.minus,
                                            IsDrained=s.IsDrained)
                    # print(f"Adding shorter version of {s.name}+{s.plus}-{s.minus} : {sig_shorter.name}+{sig_shorter.plus}-{sig_shorter.minus}")
                    self.add_signal(sig_shorter)

                        
    def update_dict_index(self):
        """Update the dictionary of all indices, and the total number of equations"""
        self.dict_index.clear()
        i=0
        for s in self.signals:
            self.dict_index[(s,"alone")]=i
            i+=1
            if s.IsDrained:
                self.dict_index[(s,"drained")]=i
                i+=1
        for d in self.drains:
            self.dict_index[(d, "alone")]=i
            i+=1
            for sig_in in self.dict_var_signals[d.input.name]:
                self.dict_index[(d, sig_in, "in")]=i
                i+=1
                self.dict_index[(d, sig_in, "ext")]=i
                i+=1
        for tmp in self.templates:
            self.dict_index[(tmp, "alone")]=i
            i+=1
            self.dict_index[(tmp, "out")]=i
            i+=1
            if tmp.irreversible:
                self.dict_index[(tmp,"load")]=i
                i+=1
            for sig_in in self.dict_var_signals[tmp.input.name]:
                self.dict_index[(tmp, sig_in, "in")]=i
                i+=1
                self.dict_index[(tmp, sig_in,"both")]=i
                i+=1
                self.dict_index[(tmp, sig_in,"ext")]=i
                i+=1
                if tmp.input.IsDrained:
                    self.dict_index[(tmp, sig_in,"in_drained")]=i
                    i+=1
        for rT in self.reporters:
            self.dict_index[(rT,"alone")]=i
            i+=1
            if rT.reversible:
                self.dict_index[(rT,"out")]=i
                i+=1
            for sig_in in self.dict_var_signals[rT.input.name]:
                self.dict_index[(rT, sig_in, "in")]=i
                i+=1
                if rT.reversible:
                    self.dict_index[(rT, sig_in, "both")]=i
                    i+=1
                self.dict_index[(rT, sig_in, "ext")]=i
                i+=1

        self.nb_equations=i
                
            
    def get_concentration(self, y, oligo, oligo2=None, option="alone"):
        assert isinstance(oligo, Signal) or isinstance(oligo, Drain) or isinstance(oligo, Template) or isinstance(oligo, Reporter)
        assert oligo2==None or isinstance(oligo2, Signal) or isinstance(oligo2, Drain) or isinstance(oligo2, Template)
        if oligo2==None:
            if option in ["alone", "out", "drained", 'load'] and (oligo, option) in self.dict_index:
                return y[self.dict_index[(oligo, option)]]
            elif option in ["ext", "in", "both", "in_drained", "drained"] and (isinstance(oligo, Template) or isinstance(oligo, Drain)):
                if oligo.input.name not in self.dict_var_signals:
                    raise Exception(f"No variable signals found for {oligo.input.name}.")
                elif len(self.dict_var_signals[oligo.input.name])==1:
                    return y[self.dict_index[(oligo, oligo.input, option)]]
                else:
                    # print(f"Getting concentration for {oligo.name} with option {option} over all variable signals.")
                    # print(f"Variable signals for {oligo.name}: {[s.name+'+'+str(s.plus)+'-'+str(s.minus) for s in self.dict_var_signals[oligo.input.name]]}")
                    return sum([y[self.dict_index[(oligo, s, option)]] for s in self.dict_var_signals[oligo.input.name]])
            else :
                raise Exception(f"Option {option} not recognized for {oligo.name}+{oligo.plus}-{oligo.minus}.")
        elif (oligo, oligo2, option) in self.dict_index:
                return y[self.dict_index[(oligo, oligo2, option)]]
        else :
            raise Exception(f"The option {option} does not exist for the complex made of {oligo.name} and {oligo2.name}+{oligo2.plus}-{oligo2.minus}")


    def update_dicts(self):
        """Update the reaction dictionaries and the kinetic dictionaries (containing association and dissociation rates k_on and k_off)"""
        self.dict_input.clear()
        self.dict_output.clear()
        self.dict_drain.clear()
        self.dict_kin_rates.clear()
        for s in self.signals:
            if not s.name in self.dict_var_signals:
                self.dict_var_signals[s.name]=[s]
            elif s not in self.dict_var_signals[s.name]:
                self.dict_var_signals[s.name].append(s) # Add the potential other versions of the signal
            for tmp in self.templates:
                if tmp.input==s and tmp.output==s:
                    if (s,tmp,'input') not in self.dict_kin_rates:
                        self.dict_kin_rates[(s,tmp,'input')]=compute_rates(s,tmp,self.temperature, self.sodium, self.magnesium, option='input')
                    if (s,tmp,'output') not in self.dict_kin_rates:
                        self.dict_kin_rates[(s,tmp,'output')]=compute_rates(s,tmp,self.temperature, self.sodium, self.magnesium, option='output')
                    if s in self.dict_output and tmp not in self.dict_output[s]:
                        self.dict_output[s].append(tmp)
                    else:
                        self.dict_output[s]=[tmp]
                    if s in self.dict_input and (tmp, 0) not in self.dict_input[s]:
                        self.dict_input[s].append((tmp,0))
                    else:
                        self.dict_input[s]=[(tmp,0)]                    
                elif tmp.output==s:
                    if (s,tmp) not in self.dict_kin_rates:
                        self.dict_kin_rates[(s,tmp)]=compute_rates(s,tmp,self.temperature, self.sodium, self.magnesium)
                    if s in self.dict_output and tmp not in self.dict_output[s]:
                        self.dict_output[s].append(tmp)
                    else:
                        self.dict_output[s]=[tmp]
                elif tmp.input==s:
                    if (s,tmp) not in self.dict_kin_rates:
                        self.dict_kin_rates[(s,tmp)]=compute_rates(s,tmp,self.temperature, self.sodium, self.magnesium)
                    if s in self.dict_input and (tmp, 0) not in self.dict_input[s]:
                        self.dict_input[s].append((tmp,0))
                    else:
                        self.dict_input[s]=[(tmp,0)]
            for d in self.drains:
                if d.input==s:
                    if (s,d) not in self.dict_kin_rates:
                        self.dict_kin_rates[(s,d)]=compute_rates(s,d,self.temperature, self.sodium, self.magnesium)
                    if s not in self.dict_drain:
                        self.dict_drain[s]=[]
                    if d not in self.dict_drain[s]:
                        self.dict_drain[s].append(d)
            for rT in self.reporters:    
                if rT.input==s or rT.output==s:
                    if (s,rT) not in self.dict_kin_rates:
                        self.dict_kin_rates[(s,rT)]=compute_rates(s,rT,self.temperature, self.sodium, self.magnesium)
        # Add the var signals in dict_input
        for s in self.signals:
            if s.name in self.dict_var_signals:
                for s_var in self.dict_var_signals[s.name]:
                    if s_var != s and s_var in self.dict_input:
                        for (tmp, _) in self.dict_input[s_var]:
                            if s not in self.dict_input:
                                self.dict_input[s]=[]
                            if (tmp,0) not in self.dict_input[s]:
                                self.dict_input[s].append((tmp,0))
                                if (s,tmp,'input') not in self.dict_kin_rates:
                                    self.dict_kin_rates[(s,tmp,'input')]=compute_rates(s,tmp,self.temperature, self.sodium, self.magnesium, option='input')
                    if s_var != s and s_var in self.dict_drain:
                        for d in self.dict_drain[s_var]:
                            if s not in self.dict_drain and s.plus==0:
                                self.dict_drain[s]=[]
                            if s.plus==0 and d not in self.dict_drain[s]:
                                self.dict_drain[s].append(d)
                                if (s,d) not in self.dict_kin_rates:
                                    self.dict_kin_rates[(s,d)]=compute_rates(s,d,self.temperature, self.sodium, self.magnesium)
            for rT in self.reporters:    
                if rT.input==s:
                    for s_var in self.dict_var_signals[s.name]:
                        if s_var != s:
                            if (s_var,rT) not in self.dict_kin_rates:
                                self.dict_kin_rates[(s_var,rT)]=compute_rates(s_var,rT,self.temperature, self.sodium, self.magnesium)

        for d in self.drains:
            if d not in self.dict_kin_rates:
                self.dict_kin_rates[d]=compute_rates(d,None,self.temperature, self.sodium, self.magnesium)
            for tmp in self.templates:
                if tmp.output==d:
                    if (d,tmp) not in self.dict_kin_rates:
                        self.dict_kin_rates[(d,tmp)]=compute_rates(d,tmp,self.temperature, self.sodium, self.magnesium)
                    if d in self.dict_output and tmp not in self.dict_output[d]:
                        self.dict_output[d].append(tmp)
                    else:
                        self.dict_output[d]=[tmp]

        # Update the coaxial stacking penalties dictionary
        for tmp in self.templates:
            if tmp not in self.dict_stack:
                self.dict_stack[tmp]=stack_slowdown(tmp, self.temperature)
        for rT in self.reporters:
            if rT not in self.dict_stack:
                self.dict_stack[rT]=stack_slowdown(rT, self.temperature)



    def init_equations(self):
        """Return the initial conditions of the equations"""
        nb_equations=self.nb_equations
        y0=[0]*nb_equations
        for s in self.signals:
            y0[self.dict_index[(s,"alone")]]=s.concentration
        for d in self.drains:
            y0[self.dict_index[(d,"alone")]]=d.concentration
        for tmp in self.templates:
            y0[self.dict_index[(tmp,"alone")]]=tmp.concentration
        for rT in self.reporters:
            y0[self.dict_index[(rT,"alone")]]=rT.concentration
        return y0


    
    def generate_equations(self,y,t):
        """Generate the equations, depending on the time step"""
        nb_equations=self.nb_equations
        equations=[0]*nb_equations
        def k_on(s1:Signal|Drain|Template, s2:Signal|Drain|Template|None, option:str="default"):
            """Return the binding rate between the signal dn its complementary"""
            if s2==None:
                return self.dict_kin_rates[d][0]
            if isinstance(s2,Template) and s2.input==s2.output:
                if option=='input':
                    return self.dict_kin_rates[(s1,s2,'input')][0]
                else:
                    return self.dict_kin_rates[(s1,s2,'output')][0]
            return self.dict_kin_rates[(s1,s2)][0]
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
        
        #Enzyme activity
        def pol(tmp_:Template|Drain):
            """Return the activity of the polymerase"""
            pol=polV(concentration=self.concentration_pol, 
                     temperature=self.temperature,
                     tmp=tmp_)
            div=1 + sum([self.get_concentration(y, tmp, option="in")/polK(temperature=self.temperature, tmp=tmp) for tmp in self.templates]) # Use of pol for "in" templates
            div+= sum([self.get_concentration(y, d, option="in")/polK(temperature=self.temperature, tmp=d) for d in self.drains]) # Use of pol for drains
            div+= sum([self.get_concentration(y, tmp, option="both")/polK_both(temperature=self.temperature, tmp=tmp) for tmp in self.templates]) # Use of pol for "both" templates 

            div+= sum([self.get_concentration(y, tmp, option="in")/polK(temperature=self.temperature, tmp=tmp) for tmp in self.dict_tmp_elongation]) # Use of pol for non-phosphorylated "in" templates
            div+= sum([self.get_concentration(y, tmp, option="both")/polK_both(temperature=self.temperature, tmp=tmp) for tmp in self.dict_tmp_elongation]) # Use of pol for  non-phosphorylated "both" templates 
            div+= sum([self.get_concentration(y, tmp, option="ext")/polK_both(temperature=self.temperature, tmp=tmp) for tmp in self.dict_tmp_elongation]) # Use of pol for  non-phosphorylated "ext" templates 
            pol/=polK(temperature=self.temperature, tmp=tmp_) * div
            return pol

        def pol_both(tmp_:Template):
            """Return the activity of the polymerase when the output is already bound"""
            pol_both=polV_both(concentration=self.concentration_pol,
                               temperature=self.temperature, 
                               tmp=tmp_)
            div=1 + sum([self.get_concentration(y, tmp, option="in")/polK(temperature=self.temperature, tmp=tmp) for tmp in self.templates]) # Use of pol for "alone" templates
            div+= sum([self.get_concentration(y, d, option="in")/polK(temperature=self.temperature, tmp=d) for d in self.drains]) # Use of pol for drains
            div+= sum([self.get_concentration(y, tmp, option="both")/polK_both(temperature=self.temperature, tmp=tmp) for tmp in self.templates]) # Use of pol for "both" templates
            pol_both/=polK_both(temperature=self.temperature, tmp=tmp_) * div
            return pol_both
        
        def BsmI(tmp_:Template):
            """Return the activity of the nicking enzyme BsmI with respect to the template"""
            enz=BsmIV(concentration=self.concentration_BsmI, 
                      temperature=self.temperature, 
                      tmp=tmp_)
            div=1 + sum([self.get_concentration(y, tmp, option="ext")/BsmIK(temperature=self.temperature, tmp=tmp) for tmp in self.templates if tmp.nick=="BsmI"]) # BsmI cleaves all the "ext" templates using it 
            enz/=BsmIK(temperature=self.temperature, tmp=tmp_) * div
            return enz
        
        def NBI(tmp_:Template):
            """Return the activity of the nicking enzyme NBI"""
            enz=NBIV(concentration=self.concentration_NBI, 
                     temperature=self.temperature, 
                     tmp=tmp_)
            div=1 + sum([self.get_concentration(y, tmp, option="ext")/NBIK(temperature=self.temperature, tmp=tmp) for tmp in self.templates if tmp.nick=="NBI"])    # NbI cleaves all the "ext" templates using it 
            enz/=NBIK(temperature=self.temperature, tmp=tmp_) * div
            return enz
        
        def exo(oligo:Signal|Drain|Template):
            """Return the activity of the exonuclease"""
            enz=exoV(concentration=self.concentration_exo, 
                     temperature=self.temperature, 
                     sig=oligo)
            div= 1 + sum([self.get_concentration(y, s, option="alone")/exoK(temperature=self.temperature, sig=s) for s in self.signals if not s.protected])          # Exonuclease degrades the non-protected signals
            div+= sum([self.get_concentration(y, d.input, option="drained")/exoK(temperature=self.temperature, sig=d.input) for d in self.drains])                   # Exonuclease degrades the drained signals 
            div+= sum([self.get_concentration(y, d, option="alone")/exoK(temperature=self.temperature, sig=d) for d in self.drains if not d.protected])              # Exonuclease degrades the non-protected drains 
            div+= sum([self.get_concentration(y, tmp, option="alone")/exoK(temperature=self.temperature, sig=tmp) for tmp in self.templates if not tmp.protected])   # Exonuclease degrades the non-protected templates
            enz/=exoK(temperature=self.temperature, sig=oligo) * div
            return enz

        def flux_in(sig_in:Signal,y):
            """Return the flux in for the signal s"""
            fluxin=0
            if sig_in in self.dict_input:
                for tmp,_ in self.dict_input[sig_in]:
                    if not tmp.irreversible:
                        fluxin+= k_off(sig_in, tmp, 'input') * self.get_concentration(y, tmp, sig_in, option="in") # The input unbinds from in
                        fluxin+= k_off(sig_in, tmp, 'input') * self.dict_stack[tmp] * self.get_concentration(y, tmp, sig_in, option="both") # The input unbinds from both
                        fluxin-= k_on(sig_in, tmp, 'input') * self.get_concentration(y, sig_in, option="alone") * self.get_concentration(y, tmp, option="alone") # The input binds to alone
                        fluxin-= k_on(sig_in, tmp, 'input') * self.get_concentration(y, sig_in, option="alone") * self.get_concentration(y, tmp, option="out") # The input binds to out
                    else:
                        fluxin+= k_off(sig_in, tmp, 'input') * self.get_concentration(y, tmp, option="load") # The input unbinds from load
                        fluxin-= k_on(sig_in,tmp,'input') * self.get_concentration(y, sig_in, option="alone") * self.get_concentration(y, tmp, option="alone") # The input binds to alone
            if sig_in.IsDrained and sig_in in self.dict_drain:
                for d in self.dict_drain[sig_in]:
                    fluxin+= k_off(sig_in,d) * self.get_concentration(y, d, sig_in, option="in") # The input unbinds from in
                    fluxin-= k_on(sig_in, d) * self.get_concentration(y, sig_in, option="alone") * (self.get_concentration(y, d, option="alone")) #The input binds to alone
            return fluxin
        
        def flux_out(s:Signal|Drain,y):
            """Return the flux out for the signal s"""
            fluxout=0
            if s in self.dict_output:
                for tmp in self.dict_output[s]:
                    fluxout+= k_off(s,tmp,'output') * self.get_concentration(y, tmp, option="out") # The output unbinds from out
                    fluxout+= k_off(s,tmp, 'output') * self.dict_stack[tmp] * self.get_concentration(y, tmp, option="both") # The output unbinds from both
                    fluxout-= k_on(s,tmp, 'output') * self.get_concentration(y, s, option="alone") * self.get_concentration(y, tmp, option="alone") # The output binds to alone
                    fluxout-= k_on(s,tmp, 'output') * self.get_concentration(y, s, option="alone") * self.get_concentration(y, tmp, option="in") # The output binds to in
                    fluxout+= pol_both(tmp) * self.get_concentration(y, tmp, option="both") # The output is displaced by the polymerase
            return fluxout

        # Signals
        for s in self.signals:
            gen=0
            gen+= flux_in(s,y)
            gen+= flux_out(s,y)
            # Case of unprotected signals
            if not s.protected:
                gen-= exo(s)*self.get_concentration(y, s, option="alone") # The exonuclease binds to the signal
            # Case of leaks
            if s in self.dict_output: # Only for produced triggers
                leak_rate=0
                for tmp in self.dict_output[s]:
                    if tmp.leak: # We use the leak rate from the template if defined
                        leak_rate+= polV(concentration=self.concentration_pol, temperature=self.temperature, tmp=tmp) * tmp.leak * self.get_concentration(y, tmp, option="alone") # Constant leak
                    elif self.leak != 0: # We use the global leak rate
                        leak_rate+= polV(concentration=self.concentration_pol, temperature=self.temperature, tmp=tmp) * self.leak * self.get_concentration(y, tmp, option="alone") # Constant leak
                    # leak_rate+= pol(tmp)* leak * self.get_concentration(y, tmp, "alone") # Leak depending on the activity of the polymerase
                gen+= leak_rate
            equations[self.dict_index[(s,"alone")]]+= gen
            #case IsDrained
            if s.plus==0 and s.IsDrained:
                gen=0
                if s in self.dict_drain:
                    for d in self.dict_drain[s]:
                        gen+= k_off(d,None) * self.get_concentration(y, d, s, option="ext") # The output unbinds from drain Template
                        gen-= k_on(d,None) * self.get_concentration(y, d, option="alone") * self.get_concentration(y, s, option="drained") # The drained rebinds to the alone drain Template
                if s in self.dict_input:
                    for tmp,_ in self.dict_input[s]:
                        gen+= k_off(tmp.input, tmp, 'input') * self.get_concentration(y, tmp, s, option="in_drained") # The drained input unbinds from in_drained
                        gen-= k_on(tmp.input, tmp, 'input') * self.get_concentration(y, s, option="drained") * self.get_concentration(y, tmp, option="alone") # The drained input binds to alone
                gen-= exo(d)* self.get_concentration(y, s, option="drained") # The exonuclease binds to drained
                equations[self.dict_index[(s,"drained")]]+= gen
            

                
        # Templates
        for tmp in self.templates:
            if tmp.nick=="BsmI":
                nick=BsmI(tmp)
            elif tmp.nick=="NBI":
                nick=NBI(tmp)

            if not tmp.irreversible:
                # case alone
                gen= k_off(tmp.output, tmp, 'output') * self.get_concentration(y, tmp, option="out") # The output unbinds from out
                for sig_in in self.dict_var_signals[tmp.input.name]:
                    gen+= k_off(sig_in, tmp, 'input') * self.get_concentration(y, tmp, sig_in, option="in") # The input unbinds from in
                    gen-= k_on(sig_in, tmp, 'input') * self.get_concentration(y, sig_in, option="alone") * self.get_concentration(y, tmp, option="alone") # The input binds to alone
                gen-= k_on(tmp.output, tmp, 'output') * self.get_concentration(y, tmp.output, option="alone") * self.get_concentration(y, tmp, option="alone") # The output binds to alone
                if not tmp.protected:
                    gen-= exo(tmp)*self.get_concentration(y, tmp, option="alone") # The exonuclease binds to alone
                if tmp.input.IsDrained and tmp.input.plus==0:
                    for sig_in in self.dict_var_signals[tmp.input.name]:
                        if sig_in.plus==0:
                            gen+= k_off(sig_in, tmp, 'input')  * self.get_concentration(y, tmp, sig_in, option="in_drained") # The drained input unbinds from in_drained
                            gen-= k_on(sig_in, tmp, 'input') * self.get_concentration(y, sig_in, option="drained") * self.get_concentration(y, tmp, option="alone") # The drained input binds to alone
                equations[self.dict_index[(tmp, "alone")]]+= gen
                #case in
                for sig_in in self.dict_var_signals[tmp.input.name]:
                    gen = k_on(sig_in, tmp, 'input') * self.get_concentration(y, sig_in, option="alone") * self.get_concentration(y, tmp, option="alone") # The input binds to alone
                    if sig_in.plus==0:
                        gen-= k_on(tmp.output, tmp, 'output') * self.get_concentration(y, tmp.output, option="alone") * self.get_concentration(y, tmp, sig_in, option="in")  # The output binds to in
                        gen+= k_off(tmp.output, tmp, 'output') * self.dict_stack[tmp] * self.get_concentration(y, tmp, sig_in, option="both")                          # The output unbinds from both
                    gen-= k_off(sig_in, tmp, 'input') * self.get_concentration(y, tmp, sig_in, option="in")  # The input unbinds from in
                    gen-= pol(tmp) * self.get_concentration(y, tmp, sig_in, option="in") # The polymerase binds to in
                    if tmp in self.dict_tmp_elongation and sig_in.minus==0: # Warning: if sig_in is able to elongate the tmp TODO
                        elongated_tmp=self.dict_tmp_elongation[tmp]
                        equations[self.dict_index[(elongated_tmp, sig_in, "in")]]+= pol(sig_in)*self.get_concentration(y, tmp, sig_in, option="in") # The polymerase bind to the non-phosphorylated template and elongates it
                        gen-= pol(sig_in)*self.get_concentration(y, tmp, sig_in, option="in")  # The polymerase bind to the non-phosphorylated template and elongates it
                    equations[self.dict_index[(tmp, sig_in, "in")]]+= gen
                    if sig_in.plus!=0:
                        sig_in_shorter=None
                        for si in self.dict_var_signals[tmp.input.name]:
                            if si.plus == 0 and si.minus == sig_in.minus:
                                sig_in_shorter=si
                                break 
                        if sig_in_shorter==None:
                            raise Exception
                        else:
                            equations[self.dict_index[(tmp, sig_in_shorter, "ext")]]+= pol(tmp) * self.get_concentration(y, tmp, sig_in, option="in") # The polymerase binds to in and creates the non-plused ext version
                #case out
                gen = k_on(tmp.output, tmp, 'output') * self.get_concentration(y, tmp.output, option="alone") * self.get_concentration(y, tmp, option="alone") # The output binds to alone
                gen-= k_off(tmp.output, tmp, 'output') * self.get_concentration(y, tmp, option="out") # The output unbinds from out
                for sig_in in self.dict_var_signals[tmp.input.name]:
                    if sig_in.plus==0:
                        gen+= k_off(sig_in, tmp, 'input') * self.dict_stack[tmp] * self.get_concentration(y, tmp, sig_in, option="both") # The input unbinds from both
                        gen-= k_on(sig_in, tmp, 'input') * self.get_concentration(y, sig_in, option="alone") * self.get_concentration(y, tmp, option="out") # The input binds to out
                equations[self.dict_index[(tmp, "out")]]+= gen
                #case both 
                for sig_in in self.dict_var_signals[tmp.input.name]:
                    if sig_in.plus!=0:
                        continue
                    gen = k_on(sig_in, tmp, 'input') * self.get_concentration(y, sig_in, option="alone") * self.get_concentration(y, tmp, option="out") # The input binds to out
                    gen+= k_on(tmp.output, tmp, 'output') * self.get_concentration(y, tmp.output, option="alone") * self.get_concentration(y, tmp, sig_in, option="in") # The output binds to in
                    gen-= k_off(sig_in, tmp, 'input') * self.dict_stack[tmp] * self.get_concentration(y, tmp, sig_in, option="both")  # The input unbinds from both
                    gen-= k_off(tmp.output, tmp, 'output') * self.dict_stack[tmp] * self.get_concentration(y, tmp, sig_in, option="both") # The output unbinds from both
                    gen-= pol_both(tmp) * self.get_concentration(y, tmp, sig_in, option="both") # The polymerase binds to both
                    gen+= nick * self.get_concentration(y, tmp, sig_in, option="ext") # The nicking enzyme binds to ext 
                    if tmp in self.dict_tmp_elongation and sig_in.minus==0: # Warning: if sig_in is able to elongate the tmp, done with a shortcut here
                        elongated_tmp=self.dict_tmp_elongation[tmp]
                        equations[self.dict_index[(elongated_tmp, sig_in, "both")]]+= pol(sig_in)*self.get_concentration(y, tmp, sig_in, option="both")   # The polymerase bind to the non-phosphorylated template and elongates it
                        gen-= pol(sig_in)*self.get_concentration(y, tmp, sig_in, option="both")                                          # The polymerase bind to the non-phosphorylated template and elongates it
                    equations[self.dict_index[(tmp, sig_in, "both")]]+= gen
                #case ext
                for sig_in in self.dict_var_signals[tmp.input.name]:
                    if sig_in.plus!=0:
                        continue
                    gen = pol(tmp) * self.get_concentration(y, tmp, sig_in, option="in") # The polymerase binds to in
                    gen+= pol_both(tmp) * self.get_concentration(y, tmp, sig_in, option="both") # The polymerase binds to both
                    gen-= nick * self.get_concentration(y, tmp, sig_in, option="ext") # The nicking enzyme binds to ext
                    if tmp in self.dict_tmp_elongation and sig_in.minus==0:
                        elongated_tmp=self.dict_tmp_elongation[tmp]
                        equations[self.dict_index[(elongated_tmp, sig_in, "ext")]]+= pol(sig_in) * self.get_concentration(y, tmp, sig_in, option="ext")   # The polymerase bind to the non-phosphorylated template and elongates it
                        gen-= pol(sig_in) * self.get_concentration(y, tmp, sig_in, option="ext")                                          # The polymerase bind to the non-phosphorylated template and elongates it
                    equations[self.dict_index[(tmp, sig_in, "ext")]]+= gen
                    #case in_drained
                    if sig_in.IsDrained:
                        if sig_in.plus!=0:
                            continue
                        gen = k_on(sig_in, tmp, 'input') * self.get_concentration(y, sig_in, option="drained") * self.get_concentration(y, tmp, option="alone") # The drained input binds to alone
                        gen-= k_off(sig_in, tmp, 'input')  * self.get_concentration(y, tmp, sig_in, option="in_drained") # The drained input unbinds from in_drained
                        equations[self.dict_index[(tmp, sig_in, "in_drained")]]+= gen
                    

            else: # If the template is a irreversible template
                ### Warning, at the moment, the irreversible templates don't have many versions of the inputs !
                # case alone
                gen = k_off(tmp.input, tmp, 'input') * self.get_concentration(y, tmp, option="load") # The input unbinds from load
                gen+= k_off(tmp.output,tmp,'output') * self.get_concentration(y, tmp, option="out") # The output unbinds from out
                gen-= k_on(tmp.input,tmp,'input') * self.get_concentration(y, tmp.input, option="alone") * self.get_concentration(y, tmp, option="alone") # The input binds to alone
                gen-= k_on(tmp.output,tmp,'output') * self.get_concentration(y, tmp.output, option="alone") * self.get_concentration(y, tmp, option="alone") # The output binds to alone
                if not tmp.protected:
                    gen-= exo(tmp)*self.get_concentration(y, tmp, option="alone") # The exonuclease binds to alone
                if tmp.input.IsDrained:
                    gen+= k_off(tmp.input,tmp,'input')  * self.get_concentration(y, tmp, option="in_drained") # The drained input unbinds from in_drained
                    gen-= k_on(tmp.input, tmp,'input') * self.get_concentration(y, tmp.input, option="drained") * self.get_concentration(y, tmp, option="alone") # The drained input binds to alone
                equations[self.dict_index[(tmp,"alone")]]+= gen
                #case load
                gen = k_on(tmp.input,tmp,'input') * self.get_concentration(y, tmp.input, option="alone") * self.get_concentration(y, tmp, option="alone") # The input binds to alone
                gen-= k_off(tmp.input, tmp, 'input') * self.get_concentration(y, tmp, option="load") # The input unbinds from load
                gen-= pol(tmp) * self.get_concentration(y, tmp, option="load") # The polymerase binds to load
                equations[self.dict_index[(tmp,"load")]]+= gen
                #case in
                gen = k_off(tmp.output,tmp,'output') * self.dict_stack[tmp] * self.get_concentration(y, tmp, option="both") # The output unbinds from both
                gen-= k_on(tmp.output,tmp,'output') * self.get_concentration(y, tmp, option="in") * self.get_concentration(y, tmp.output, option="alone") # The output binds to in
                gen-= pol(tmp) * self.get_concentration(y, tmp, option="in") # The polymerase binds to in
                equations[self.dict_index[(tmp, tmp.input, "in")]]+= gen
                #case out
                gen = k_on(tmp.output,tmp,'output') * self.get_concentration(y, tmp.output, option="alone") * self.get_concentration(y, tmp, option="alone") # The output binds to alone
                gen-= k_off(tmp.output,tmp,'output') * self.get_concentration(y, tmp, option="out")  # The output unbinds from out
                equations[self.dict_index[(tmp,"out")]]+= gen
                #case both
                gen = k_on(tmp.output,tmp,'output') * self.get_concentration(y, tmp.output, option="alone") * self.get_concentration(y, tmp, option="in") # The output binds to in
                gen-= k_off(tmp.output,tmp,'output') * self.dict_stack[tmp] * self.get_concentration(y, tmp, option="both") # The output unbinds from both
                gen+= nick*self.get_concentration(y, tmp, option="ext") # The nicking enzyme binds to ext
                gen-= pol_both(tmp) * self.get_concentration(y, tmp, option="both") # The polymerase binds to both
                equations[self.dict_index[(tmp, tmp.input, "both")]]+= gen
                #case ext
                gen = pol(tmp) * self.get_concentration(y, tmp, option="in") # The polymerase binds to in
                gen+= pol(tmp) * self.get_concentration(y, tmp, option="load") # The polymerase binds to load
                gen+= pol_both(tmp) * self.get_concentration(y, tmp, option="both") # The polymerase binds to both
                gen-= nick*self.get_concentration(y, tmp, option="ext") # The nicking enzyme binds to ext
                equations[self.dict_index[(tmp, tmp.input, "ext")]]+= gen
                #case in_drained
                if tmp.input.IsDrained:
                    gen = k_on(tmp.input, tmp, 'input') * self.get_concentration(y, tmp.input, option="drained") * self.get_concentration(y, tmp, option="alone") # The drained input binds to alone
                    gen-= k_off(tmp.input,tmp, 'input')  * self.get_concentration(y, tmp, option="in_drained") # The drained input unbinds from in_drained
                    equations[self.dict_index[(tmp, tmp.input, "in_drained")]]+= gen
                    

        #drains
        for d in self.drains:
            #case alone
            gen = 0
            for sig_in in self.dict_var_signals[d.input.name]:
                if sig_in.plus==0:
                    gen+= k_off(d, None) * self.get_concentration(y, d, sig_in, option="ext") # The drained signal unbinds from the pseudoTemplate
                    gen+= k_off(sig_in, d) * self.get_concentration(y, d, sig_in, option="in") # The input unbinds from in
                    gen-= k_on(sig_in, d) * self.get_concentration(y, sig_in, option="alone") * self.get_concentration(y, d, option="alone") # The input binds to alone
                    gen-= k_on(d, None) * self.get_concentration(y, d, option="alone") * self.get_concentration(y, sig_in, option="drained") # The drained signal rebinds to alone
            gen+= flux_out(d, y) # The created drains
            if not d.protected:
                gen-= exo(d) * self.get_concentration(y, d, option="alone") # The exonuclease binds to alone
            if d in self.dict_output :
                leak_rate=0
                for tmp in self.dict_output[d]:
                    if tmp.leak: # We use the leak rate from the template if defined
                        leak_rate+= polV(concentration=self.concentration_pol, temperature=self.temperature, tmp=tmp) * tmp.leak * self.get_concentration(y, tmp, option="alone") # Constant leak
                    elif self.leak != 0: # We use the global leak rate
                        leak_rate+= polV(concentration=self.concentration_pol, temperature=self.temperature, tmp=tmp) * self.leak * self.get_concentration(y, tmp, option="alone") # Constant leak
                    # leak_rate+= pol(tmp)* leak * self.get_concentration(y, tmp, "alone") # Leak depending on the activity of the polymerase
                gen+= leak_rate
            equations[self.dict_index[(d,"alone")]]+= gen
            #case in
            for sig_in in self.dict_var_signals[d.input.name]:
                if sig_in.plus==0:
                    gen = k_on(sig_in, d) * self.get_concentration(y, sig_in, option="alone") * self.get_concentration(y, d, option="alone") # The input binds to alone
                    gen-= k_off(sig_in, d) * self.get_concentration(y, d, sig_in, option="in") # The input unbinds from in
                    gen-= pol(d) * self.get_concentration(y, d, sig_in, option="in") # The polymerase binds to in
                    equations[self.dict_index[(d, sig_in, "in")]]+= gen
            #case ext
            for sig_in in self.dict_var_signals[d.input.name]:
                if sig_in.plus==0:
                    gen = k_on(d, None) * self.get_concentration(y, d, option="alone") * self.get_concentration(y, sig_in, option="drained") # The drained rebinds to alone
                    gen-= k_off(d, None) * self.get_concentration(y, d, sig_in, option="ext") # The drained signal unbinds from the pT
                    gen+= pol(d) * self.get_concentration(y, d, sig_in, option="in") # The polymerase binds to in
                    equations[self.dict_index[(d, sig_in, "ext")]]+= gen


        # Reporter templates
        for rT in self.reporters:
            if rT.reversible:
                # case alone
                gen = k_off(rT.output, rT, 'output') * self.get_concentration(y, rT, option="out") # The output unbinds from out
                for sig_in in self.dict_var_signals[rT.input.name]:
                    gen+= k_off(sig_in, rT, 'input') * self.get_concentration(y, rT, sig_in, option="in") # The input unbinds from in
                    gen-= k_on(sig_in, rT, 'input') * self.get_concentration(y, sig_in, option="alone") * self.get_concentration(y, rT, option="alone") # The input binds to alone
                gen-= k_on(rT.output, rT, 'output') * self.get_concentration(y, rT.output, option="alone") * self.get_concentration(y, rT, option="alone") # The output binds to alone
                equations[self.dict_index[(rT, "alone")]]+= gen
                #case in
                for sig_in in self.dict_var_signals[rT.input.name]:
                    gen = k_on(sig_in, rT, 'input') * self.get_concentration(y, sig_in, option="alone") * self.get_concentration(y, rT, option="alone") # The input binds to alone
                    gen+= k_off(rT.output, rT, 'output') * self.dict_stack[tmp] * self.get_concentration(y, rT, sig_in, option="both")                            # The output unbinds from both
                    gen-= k_on(rT.output, rT, 'output') * self.get_concentration(y, rT.output, option="alone") * self.get_concentration(y, rT, sig_in, option="in") # The output binds to in
                    gen-= k_off(sig_in, rT, 'input') * self.get_concentration(y, rT, sig_in, option="in")  # The input unbinds from in
                    gen-= pol(rT) * self.get_concentration(y, rT, sig_in, option="in") # The polymerase binds to in
                    equations[self.dict_index[(rT, sig_in, "in")]]+= gen
                #case out
                gen = k_on(rT.output, rT, 'output') * self.get_concentration(y, rT.output, option="alone") * self.get_concentration(y, rT, option="alone") # The output binds to alone
                gen-= k_off(rT.output, rT, 'output') * self.get_concentration(y, rT, option="out") # The output unbinds from out
                for sig_in in self.dict_var_signals[rT.input.name]:
                    gen+= k_off(sig_in, rT, 'input') * self.dict_stack[tmp] * self.get_concentration(y, rT, sig_in, option="both") # The input unbinds from both
                    gen-= k_on(sig_in, rT, 'input') * self.get_concentration(y, sig_in, option="alone") * self.get_concentration(y, rT, option="out") # The input binds to out
                equations[self.dict_index[(rT, "out")]]+= gen
                #case both 
                for sig_in in self.dict_var_signals[rT.input.name]:
                    gen = k_on(sig_in, rT, 'input') * self.get_concentration(y, sig_in, option="alone") * self.get_concentration(y, rT, option="out") # The input binds to out
                    gen+= k_on(rT.output, rT, 'output') * self.get_concentration(y, rT.output, option="alone") * self.get_concentration(y, rT, sig_in, option="in") # The output binds to in
                    gen-= k_off(sig_in, rT, 'input') * self.dict_stack[tmp] * self.get_concentration(y, rT, sig_in, option="both")  # The input unbinds from both
                    gen-= k_off(rT.output, rT, 'output') * self.dict_stack[tmp] * self.get_concentration(y, rT, sig_in, option="both") # The output unbinds from both
                    gen-= pol_both(rT) * self.get_concentration(y, rT, sig_in, option="both") # The polymerase binds to both
                    # Nicking enzyme
                    gen+= nick * self.get_concentration(y, rT, sig_in, option="ext") # The nicking enzyme binds to ext 
                    equations[self.dict_index[(rT, sig_in, "both")]]+= gen
                #case ext
                for sig_in in self.dict_var_signals[rT.input.name]:
                    gen = pol(rT) * self.get_concentration(y, rT, sig_in, option="in") # The polymerase binds to in
                    gen+= pol_both(rT) * self.get_concentration(y, rT, sig_in, option="both") # The polymerase binds to both
                    gen-= nick * self.get_concentration(y, rT, sig_in, option="ext") # The nicking enzyme binds to ext
                    equations[self.dict_index[(rT, sig_in, "ext")]]+= gen

                # input of the reporter
                for sig_in in self.dict_var_signals[rT.input.name]:
                    gen = k_off(sig_in, rT, 'input') * self.get_concentration(y, rT, sig_in, option="in") # The input unbinds from in
                    gen-= k_on(sig_in, rT, 'input') * self.get_concentration(y, sig_in, option="alone") * self.get_concentration(y, rT, option="alone") # The input binds to alone
                    gen-= k_on(sig_in, rT, 'input') * self.get_concentration(y, sig_in, option="alone") * self.get_concentration(y, rT, option="out") # The input binds to out
                    gen+= k_off(sig_in, rT, 'input') * self.dict_stack[tmp] * self.get_concentration(y, rT, sig_in, option="both") # The input unbinds from both
                    equations[self.dict_index[(sig_in, "alone")]]+= gen

                # output of the reporter
                gen = k_off(rT.output, rT, 'output') * self.get_concentration(y, rT, option="out") # The output unbinds from out
                gen-= k_on(rT.output, rT, 'output') * self.get_concentration(y, rT.output, option="alone") * self.get_concentration(y, rT, option="alone") # The output binds to alone
                for sig_in in self.dict_var_signals[rT.input.name]:
                    gen-= k_on(rT.output, rT, 'output') * self.get_concentration(y, rT.output, option="alone") * self.get_concentration(y, rT, sig_in, option="in")  # The output binds to in
                    gen+= k_off(rT.output, rT, 'output') * self.dict_stack[tmp] * self.get_concentration(y, rT, sig_in, option="both") # The output unbinds from both
                equations[self.dict_index[(rT.output, "alone")]]+= gen


        return equations

    def solve_system(self, t:List[float], y0:List[float]|None=None):
        """Solve the system of equations"""
        self.add_missing_oligos()
        self.update_dicts()
        self.update_dict_index()
        # self.associate_index()
        if y0 is None:
            y0=self.init_equations()
        fun=lambda y,t : self.generate_equations(y,t)
        y = odeint(fun, 
                   y0, 
                   t,
                #    rtol=1e-8, 
                #    atol=1e-12,
                   )
        return y
    

    def concentration_list(self, y, name: str, option: str = "alone", oligo2=None, plus=0, minus=0):
        """
        Provide the concentration list of a signal, drain, or template for a specific option, for Evagreen plotting.

        Args:
            y: The solution array from ODE integration.
            name: The name of the signal, drain, or template.
            option: The state option (e.g., "alone", "in", "out", "both", "ext", "drained").
            oligo2: Optional second oligo for complex states (e.g., (template, signal, "in")).

        Returns:
            The concentration list for the specified oligo and option.
        """
        
        if name == "Evagreen":
            # Calculate total concentration of all non-alone strands for Evagreen plotting
            concentrations = np.zeros_like(y[:, 0], dtype=float)
            for d in self.drains:
                for sig_in in self.dict_var_signals[d.input.name]:
                    concentrations += y[:, self.dict_index[(d, sig_in, "in")]] * len(d.input.sequence)  # in
                    concentrations += y[:, self.dict_index[(d, sig_in, "ext")]] * len(d.sequence)  # ext
            for tmp in self.templates:
                concentrations += y[:, self.dict_index[(tmp, "out")]] * len(tmp.output.sequence)  # out
                if tmp.irreversible:
                    concentrations += y[:, self.dict_index[(tmp, "load")]] * (len(tmp.sequence) - len(tmp.output.sequence))  # load
                for sig_in in self.dict_var_signals[tmp.input.name]:
                    concentrations += y[:, self.dict_index[(tmp, sig_in, "in")]] * (len(tmp.sequence) - len(tmp.output.sequence))  # in
                    concentrations += y[:, self.dict_index[(tmp, sig_in, "both")]] * len(tmp.sequence)  # both
                    concentrations += y[:, self.dict_index[(tmp, sig_in, "ext")]] * len(tmp.sequence)  # ext
                    if tmp.input.IsDrained:
                        concentrations += y[:, self.dict_index[(tmp, sig_in, "in_drained")]] * (len(tmp.sequence) - len(tmp.output.sequence))  # in_drained
            for rT in self.reporters:
                concentrations += y[:, self.dict_index[(rT, "out")]] * len(rT.output.sequence)  # out
                if not rT.reversible:
                    concentrations += y[:, self.dict_index[(rT, "load")]] * (len(rT.sequence) - len(rT.output.sequence))  # load
                for sig_in in self.dict_var_signals[rT.input.name]:
                    concentrations += y[:, self.dict_index[(rT, sig_in, "in")]] * (len(rT.sequence) - len(rT.output.sequence))  # in
                    concentrations += y[:, self.dict_index[(rT, sig_in, "both")]] * len(rT.sequence)  # both
                    concentrations += y[:, self.dict_index[(rT, sig_in, "ext")]] * len(rT.sequence)  # ext
            return concentrations / np.max(concentrations)


        oligo = None
        for s in self.signals:
                if s.name == name and s.plus== plus and s.minus==minus:
                    oligo = s
                    break
        if oligo is None:
            for d in self.drains:
                if d.name == name:
                    oligo = d
                    break
        if oligo is None:
            for tmp in self.templates:
                if tmp.name == name:
                    oligo = tmp
                    break
        if oligo is None:
            for rT in self.reporters:
                if rT.name == name:
                    oligo = rT
                    break
        if oligo is None:
            raise Exception(f"No oligo found with name: {name}")

        # Handle specific options for signals, drains, and templates
        if oligo2 is None:
            if option in ["alone", "out", "drained"] and (oligo, option) in self.dict_index:
                return y[:, self.dict_index[(oligo, option)]]
            elif option in ["ext", "in", "both", "in_drained"] and (isinstance(oligo, Template) or isinstance(oligo, Drain)):
                if isinstance(oligo, Drain):
                    return np.sum([y[:, self.dict_index[(oligo, s, option)]] for s in self.dict_var_signals[oligo.input.name]], axis=0)
                elif isinstance(oligo, Template):
                    return np.sum([y[:, self.dict_index[(oligo, s, option)]] for s in self.dict_var_signals[oligo.input.name]], axis=0)            
            # Handle the "all" option for templates
            elif option == "all":
                concentrations = np.zeros_like(y[:, 0], dtype=float)
                for tmp in self.templates:
                    if tmp.name == name:
                        concentrations+= y[:, self.dict_index[(tmp, "out")]]
                        concentrations+= y[:, self.dict_index[(tmp, "alone")]]
                        if tmp.irreversible:
                            concentrations+= y[:,self.dict_index[(tmp, "load")]]
                        for sig_in in self.dict_var_signals[tmp.input.name]:
                            concentrations+= y[:, self.dict_index[(tmp, sig_in, "in")]]
                            concentrations+= y[:, self.dict_index[(tmp, sig_in, "both")]]
                            concentrations+= y[:, self.dict_index[(tmp, sig_in, "ext")]]
                            if tmp.input.IsDrained:
                                concentrations+= y[:, self.dict_index[(tmp, sig_in, "in_drained")]]
                for d in self.drains:
                    if d.name == name:
                        concentrations+= y[:, self.dict_index[(d, "alone")]]
                        for sig_in in self.dict_var_signals[d.input.name]:
                            concentrations+= y[:, self.dict_index[(d, sig_in, "in")]]
                            concentrations+= y[:, self.dict_index[(d, sig_in, "ext")]]
                return concentrations
            else:
                raise Exception(f"Option '{option}' not recognized for oligo: {oligo.name}")

        else:
            # Handle complex states (e.g., (template, signal, "in"))
            if (oligo, oligo2, option) in self.dict_index:
                return y[:, self.dict_index[(oligo, oligo2, option)]]
            # Handle the "all" option for templates
            if option == "all":
                raise Exception("The 'all' option is not supported for complexes involving two oligos.")
            else:
                raise Exception(f"Option '{option}' does not exist for the complex made of {oligo.name} and {oligo2.name}")
            
        
    def fluorescence_list(self, y, rT_name):
        conc = np.zeros_like(y[:, 0], dtype=float)
        for rT in self.reporters:
            if rT.name == rT_name:
                conc += y[:, self.dict_index[(rT, "out")]]
                for sig_in in self.dict_var_signals[rT.input.name]:
                    conc += y[:, self.dict_index[(rT, sig_in, "in")]]
                    conc += y[:, self.dict_index[(rT, sig_in, "both")]]
                    conc += y[:, self.dict_index[(rT, sig_in, "ext")]]
                return conc
        raise Exception(f"No reporter template found with name: {rT_name}")