from strands import Signal, Drain, Template, Reporter, reverse_complement
from nupack import Model, Strand, Complex, Tube, SetSpec, tube_analysis
import math



def find_subsequence(seq1,seq2):
    """Find the longest subsequence along the one the two strands bind"""
    def is_complement(a,b):
        if (a,b)==('A','T') or (a,b)==('T','A') or (a,b)==('G','C') or (a,b)==('C','G') :
            return True
        else:
            return False
    
    if len(seq1) > len(seq2):
        seq1,seq2=seq2,seq1
    seq2=seq2[::-1]
    
    max_length=0
    max_seq=''
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            length = 0
            while (i + length < len(seq1)) and (j + length < len(seq2)):
                if is_complement(seq1[i + length], seq2[j + length]):
                    length += 1
                else:
                    break
            if length > max_length:
                max_length = length
                max_seq = seq1[i:i + length]
    return max_seq



def compute_kon_NN(seq,temp):
    """Computes the association rate k_on between a DNA strand and its complementary strand, with the nearest neighbor model described by Rejali et al. 
    in the case of 1M NaCl.
    The result is given in M-1.s-1 units."""
    kb=1.3e-23
    h=6.6e-34
    R=8.314
    temp+=273 # °C -> K

    dict_H={
    'AA':0,
    'TT':0,
    'AT':0,
    'TA':0,
    'CA':0,
    'TG':0,
    'GT':0,
    'AC':0,
    'CT':0,
    'AG':0,
    'GA':0,
    'TC':0,
    'CG':0,
    'GC':0,
    'GG':0,
    'CC':0,
    'init':0,
    'term':0
    }

    dict_S={
    'AA':-0.08,
    'TT':-0.08,
    'AT':-0.16,
    'TA':-0.41,
    'CA':-0.35,
    'TG':-0.35,
    'GT':-0.02,
    'AC':-0.02,
    'CT':-0.29,
    'AG':-0.29,
    'GA':-0.22,
    'TC':-0.22,
    'CG':0.08,
    'GC':-0.02,
    'GG':0.47,
    'CC':0.47,
    'init':-25.1,
    'term':0.32
    }

    dH=dict_H['init']
    dS=dict_S['init']
    for i in range (len(seq)-1):
        couple=seq[i:i+2]
        dH+=dict_H[couple]
        dS+=dict_S[couple]
    if seq[0]=='A' or seq[0]=='T':
        dH+=dict_H['term']
        dS+=dict_S['term']
    if seq[-1]=='A' or seq[-1]=='T':
        dH+=dict_H['term']
        dS+=dict_S['term']
    dH *= 4184  # kcal/mol → J/mol
    dS *= 4.184  # cal/mol·K → J/mol·K
    dG=dH - temp*dS
    kon=(kb * temp / h) * math.exp(-dG/(R*temp))
    return kon

def compute_Kd(seqA, seqB, concA, concB, temp_celsius=37.0, sodium=0.05, magnesium=0.012):
    """
    Computes the dissociation constant K_D = [A][B]/[AB] at equilibrium.

    Parameters:
        seqA (str): DNA sequence A.
        seqB (str): DNA sequence B (not necessarily the perfect complement).
        concA (float): Initial concentration of strand A (M).
        concB (float): Initial concentration of strand B (M).
        temp_celsius (float): Temperature in Celsius.
        sodium (float): Sodium ion concentration in M.
        magnesium (float): Magnesium ion concentration in M.

    Returns:
        float: Estimated K_D in M.
    """
    # Set up and call NUPACK model
    model = Model(material='dna', celsius=temp_celsius, sodium=sodium, magnesium=magnesium)
    A = Strand(seqA, name='A')
    B = Strand(seqB, name='B')
    duplex = Complex([A, B], name='Duplex')
    tube = Tube(strands={A: concA, B: concB}, complexes=SetSpec(max_size=2), name='Tube')
    result = tube_analysis(tubes=[tube], model=model)
    
    # Extract equilibrium concentrations
    conc_duplex = result.tubes[tube].complex_concentrations.get(duplex, 0.0)
    freeA = concA - conc_duplex
    freeB = concB - conc_duplex

    # Compute K_D
    if conc_duplex > 0:
        KD = (freeA * freeB) / conc_duplex
    else:
        KD = float('inf')  # Essentially no binding
    return KD


def compute_koff_from_kon(seq,temp, sodium, magnesium):
    """Computes the dissociation rate k_off between a DNA strand and its complementary strand, thanks to k_on and K_d.
    In s-1 units"""
    K_d=compute_Kd(seq, reverse_complement(seq),1e-8,1e-8,temp_celsius=temp, sodium=sodium, magnesium=magnesium)
    k_on=compute_kon_NN(seq,temp)
    k_off=k_on*K_d
    return k_off


def compute_rates(s:Signal|Drain|Template, tmp:Signal|Drain|Template|None, temperature:float, sodium:float, magnesium:float, option:str="default"):
    """Computes the rates of the association reaction between two oligos.
    If tmp is None, computes the rates of an oligo with its complementary strand.
    The result is given in nM-1 min-1 for k_on and min-1 for k_off."""
    if tmp==None:
        binding_sequence=s.sequence
    elif option=='input':
        seq1=s.sequence
        seq2=tmp.sequence
        if len(seq1) > len(seq2):
            seq1,seq2=seq2,seq1
        seq2=seq2[-len(seq1)-4:]
        binding_sequence = find_subsequence(seq1,seq2)
    else:
        binding_sequence = find_subsequence(s.sequence,tmp.sequence)
    K_d=compute_Kd(binding_sequence, reverse_complement(binding_sequence),1e-8,1e-8,temp_celsius=temperature, sodium=sodium, magnesium=magnesium)
    k_on=compute_kon_NN(binding_sequence,temperature)
    k_off=k_on*K_d

    k_on = k_on * 1e-9 * 60 # Convert from M-1 s-1 to nM-1 min-1
    k_off = k_off * 60 # Convert from s-1 to min-1
    return k_on, k_off


# Enzyme kinetics functions

# Polymerase, without strand displacement
def polV(concentration:float=None, temperature:float=None, tmp:Template=None|Drain):
    """
    Return the maximum speed of the polymerase with respect to template in nM/min
    concentration: concentration of polymerase (100% = 2.000 U/µL)
    Reference value is 25.6 U/mL.
    temperature: temperature in °C
    tmp: template or drain strand (for latter, in case of sequence dependence)
    """
    if concentration is not None:
        return concentration * 41.1 # From Montagne2011 and Padirac2012 (at 42°C), but divided by 2 according to DACCAD
    return 1050 # Half value of Padirac2012 (as in DACCAD)
        
def polK(temperature:float=None, tmp:Template|Drain=None):
    """
    Return the Michaelis constant of the polymerase with respect to template in nM
    temperature: temperature in °C
    tmp: template or drain strand (for latter, in case of sequence dependence)
    """
    return 80 # Value of Padirac2012 (as in DACCAD) (at 42°C)
        
# Polymerase, with strand displacement
def polV_both(concentration:float=None, temperature:float=None, tmp:Template=None):
    """
    Return the maximum speed of the polymerase with respect to template when the output is already bound  in nM/min
    concentration:  concentration of polymerase (100% = 2.000 U/µL)
    Reference value is 25.6 U/mL.
    temperature: temperature in °C
    tmp: template or drain strand (for latter, in case of sequence dependence)
    """
    if concentration is not None:
        return concentration * 8.2 # From Montagne2011 and Padirac2012 (at 42°C), but divided by 2 according to DACCAD
    return 210 # Half value of Padirac2012 (Warning: no differences in DACCAD) (at 42°C)

def polK_both(temperature:float=None, tmp:Template=None):
    """
    Return the Michaelis constant of the polymerase with respect to template when the output is already bound in nM
    temperature: temperature in °C
    tmp: template or drain strand (for latter, in case of sequence dependence)
    """
    return 5.5 # Value of Padirac2012 (as in DACCAD) (at 42°C)
        

# Nb.BsmI nicking enzyme
def BsmIV(concentration:float=None, temperature:float=None, tmp:Template=None):
    """
    Return the maximum speed of the Nb.BsmI enzyme with respect to template in nM/min
    concentration: concentration of Nb.BsmI (100% = 10.000 U/µL)
    Reference value is 10 U/mL (but could be 100 nM to be ~ NBI).
    temperature: temperature in °C
    tmp: template or drain strand (for latter, in case of sequence dependence)
    """
    if concentration is not None:
        return concentration * 0.19 # Value from Montagne2016
    return 1.9 # Value from Montagne2016  (at 45°C)
        
def BsmIK(temperature:float=None, tmp:Template=None):
    """
    Return the Michaelis constant of the Nb.BsmI enzyme with respect to template in nM
    temperature: temperature in °C
    tmp: template or drain strand (for latter, in case of sequence dependence)
    """
    return 9 # Value from Montagne2016 (at 45°C)
        

# Nt.BstNBI nicking enzyme
def NBIV(concentration:float=None, temperature:float=None, tmp:Template=None):
    """
    Return the maximum speed of the Nt.BstNBI enzyme with respect to template in nM/min
    concentration:  concentration of NBI (100% = 10.000 U/µL)
    reference value is 50 U/mL
    temperature: temperature in °C
    tmp: template or drain strand (for latter, in case of sequence dependence)
    """
    if concentration is not None:
        return concentration * 1.8 # # Guessed from Padirac2012
    return 80 # Value of Padirac2012 (as in DACCAD) (at 42°C)
        
def NBIK(temperature:float=None, tmp:Template=None):
    """
    Return the Michaelis constant of the NBI enzyme with respect to template in nM
    temperature: temperature in °C
    tmp: template or drain strand (for latter, in case of sequence dependence)
    """
    return 30 # Value of Padirac2012 (as in DACCAD) (at 42°C)
        

# Exonuclease
def exoV(concentration:float=None, temperature:float=None, sig:Signal|Drain=None):
    """
    Return the maximum speed of the exonuclease with respect to signal in nM/min
    concentration:  concentration of ttRecJ 
    reference value is 50 nM
    (Old one: 100% = ttRecJ/140 = 3.34 µM
    New one: 100% = 30 nM)
    temperature: temperature in °C
    tmp: template or drain strand (for latter, in case of sequence dependence)
    """
    if concentration is not None:
        return concentration * 6 # Value from Padirac2012 (at 42°C)
    return 300 # Value of Padirac2012 (as in DACCAD) (at 42°C)
        
def exoK(temperature:float=None, sig:Signal|Drain=None):
    """
    Return the Michaelis constant of the exonuclease with respect to signal in nM
    temperature: temperature in °C
    tmp: template or drain strand (for latter, in case of sequence dependence)
    """
    return 440 # Value of Padirac (as in DACCAD)  (WARNING: different values for signals and inhibitors, need some investigation here)
    # 440 for 11bp long strands (at 42°C)
    # 150 for 15bp long strands (at 42°C) 
    # -> Errors up to 3x depending on the strand length, at low substrate concentrations.



stack=0.035 # Penalty for dissociation due to stacking between the input and output signals when both are bound to the template

def stack_slowdown(tmp:Template|Reporter,temperature:float=42.0):
    # Constants
    R = 1.987e-3  # Gas constant in kcal/mol·K
    T=temperature+273.15

    # Data from Punnoose et al. (2023)
    stack_energy = {
        "GA" : -2.3,
        "AG" : -2.3,
        "AA" : -2.3,
        "GG" : -2.3,
        "GC" : -2.1,
        "CG" : -2.1,
        "AC" : -1.9,
        "CA" : -1.9,
        "GT":-1.7,
        "TG":-1.7,
        "AT":-1.5,
        "TA":-1.5,
        "TT":-0.8,
        "CC":-0.6,
        "CT":-0.5,
        "TC":-0.5,
    }
        

    tmp_seq=tmp.sequence
    seq_out=tmp.output.sequence

    if not tmp_seq or not seq_out:
        return 0.035
    else:
        nick_seq = seq_out[0]+ reverse_complement(tmp_seq[len(seq_out)])
        stack_energy_nick = stack_energy[nick_seq]
        return math.exp(stack_energy_nick / (R * T))
