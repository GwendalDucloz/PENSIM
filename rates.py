from strands import Signal, Drain, Template
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


def compute_koff(seq,temp):
    """Computes the dissociation rate k_off between a DNA strand and its complementary strand, with the nearest neighbor model described by Rejali et al. 
    This model doesn't depend on slat concentrations at the moment."""
    kb=1.3e-23
    h=6.6e-34
    R=8.314
    temp+=273 # °C -> K

    dict_H={
    'AA':9.2,
    'TT':9.2,
    'AT':8.6,
    'TA':5.6,
    'CA':11.9,
    'TG':11.9,
    'GT':9.6,
    'AC':9.6,
    'CT':10.2,
    'AG':10.2,
    'GA':8.1,
    'TC':8.1,
    'CG':14.5,
    'GC':11.2,
    'GG':9.8,
    'CC':9.8,
    'init':-14.8,
    'term':-1
    }

    dict_S={
    'AA':26.5,
    'TT':26.5,
    'AT':24.2,
    'TA':15.8,
    'CA':33.5,
    'TG':33.5,
    'GT':25.9,
    'AC':25.9,
    'CT':28.9,
    'AG':28.9,
    'GA':21.7,
    'TC':21.7,
    'CG':40.7,
    'GC':29.2,
    'GG':26.6,
    'CC':26.6,
    'init':-66.8,
    'term':-2.3
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
    koff=(kb * temp / h) * math.exp(-dG/(R*temp))
    return '{:.2e}'.format(koff)

def reverse_complement(seq):
    """
    Returns the reverse complement of a DNA sequence.
    """
    complement = str.maketrans("ATCGatcg", "TAGCtagc")
    return seq.translate(complement)[::-1]

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


def compute_kon(seq,temp, sodium, magnesium):
    """Computes the association rate k_on between a DNA strand and its complementary strand, thanks to k_off and K_d"""
    K_d=compute_Kd(seq, reverse_complement(seq),1e-8,1e-8,temp_celsius=temp, sodium=sodium, magnesium=magnesium)
    k_off=compute_koff(seq,temp)
    k_on=float(k_off)/K_d
    return '{:.2e}'.format(k_on)

def compute_rates(s:Signal|Drain|Template, tmp:Signal|Drain|Template|None, temperature:float, sodium:float, magnesium:float, option:str="default"):
    """Computes the rates of the association reaction between the signal and the template"""
    if tmp==None:
        binding_sequence=s.sequence
    elif option=='input':
        seq1=s.sequence
        seq2=tmp.sequence
        if len(seq1) > len(seq2):
            seq1,seq2=seq2,seq1
        seq2=seq2[len(seq1):]
        binding_sequence = find_subsequence(seq1,seq2)
    else:
        binding_sequence = find_subsequence(s.sequence,tmp.sequence)
    K_d=compute_Kd(binding_sequence, reverse_complement(binding_sequence),1e-8,1e-8,temp_celsius=temperature, sodium=sodium, magnesium=magnesium)
    k_off=float(compute_koff(binding_sequence,temperature))
    k_on=k_off/K_d
    return k_on, k_off


def polV(tmp:Template|Drain):
    """Return the maximum speed of the polymerase with respect to template"""
    return 1050/60 # Half value of Padirac (as in DACCAD)
        
def polK(tmp:Template|Drain):
    """Return the Michaelis constant of the polymerase with respect to template"""
    return 80 # Value of Padirac (as in DACCAD)
        
def polV_both(tmp:Template):
    """Return the maximum speed of the polymerase with respect to template when the output is already bound"""
    return 210/60 # Half value of Padirac (Warning: no differences in DACCAD)

def polK_both(tmp:Template):
    """Return the Michaelis constant of the polymerase with respect to template when the output is already bound"""
    return 5.5 # Value of Padirac (as in DACCAD)
        
def bsmIV(tmp:Template):
    """Return the maximum speed of the bsmI enzyme with respect to template"""
    return 1.9/60 # Value from Montagne et al.
    # return 19/60 # Cheat
        
def bsmIK(tmp:Template):
    """Return the Michaelis constant of the bsmI enzyme with respect to template"""
    return 9 # Value from Montagne et al. 
        
def nbIV(tmp:Template):
    """Return the maximum speed of the nbI enzyme with respect to template"""
    return 80/60 # Value of Padirac (as in DACCAD)
        
def nbIK(tmp:Template):
    """Return the Michaelis constant of the nbI enzyme with respect to template"""
    return 30 # Value of Padirac (as in DACCAD)
        
def exoV(sig:Signal|Drain):
    """Return the maximum speed of the exonuclease with respect to signal"""
    return 300/60 # Value of Padirac (as in DACCAD)
        
def exoK(sig:Signal|Drain):
    """Return the Michaelis constant of the exonuclease with respect to signal"""
    return 440 # Value of Padirac (as in DACCAD)  (WARNING: different values for signals and inhibitors, need some investigation here)

stack=0.2 # Penalty for dissociation due to stacking between the input and output signals when both are bound to the template