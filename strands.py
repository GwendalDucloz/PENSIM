class Signal:
    def __init__(self,
                 name:str, 
                 concentration:float = 0, 
                 sequence:str = None, 
                 IsDrained:bool =False,
                 plus:int = 0,
                 minus:int =0,
                 protected:bool=False
                 ):
        self.name=name
        self.sequence=sequence
        self.concentration=concentration
        self.IsDrained=IsDrained
        self.plus=plus
        self.minus=minus
        self.protected=protected
        
    def print(self):
        if self.plus and self.minus:
            print(f"Signal: {self.name}+{self.plus}-{self.minus}")
        elif self.plus:
            print(f"Signal: {self.name}+{self.plus}")
        elif self.minus:
            print(f"Signal: {self.name}-{self.minus}")
        else:
            print(f"Signal: {self.name}")
        print(f"sequence: {self.sequence}")
        print(f"concentration: {self.concentration} nM")
        print(f"protected: {self.protected}")
        print(f"is drained: {self.IsDrained}")

class Drain:
    def __init__(self,
                 name:str,
                 input:Signal,
                 concentration:float=0,
                 sequence:str|None=None,
                 protected:bool=False):
        self.name=name
        self.input=input
        self.concentration = concentration
        self.sequence=sequence
        self.protected=protected

        

    def print(self):
        print(f"Drain: {self.name}")
        print(f"concentration: {self.concentration} nM")
        print(f"Signal selfed: {self.input.name}")
        print(f"sequence: {self.sequence}")
        print(f"protected: {self.protected}")


class Template:
    def __init__(self, 
                 name:str, 
                 input:Signal, 
                 output:Signal|Drain, 
                 concentration:float = 0, 
                 sequence:str = None, 
                 protected:bool=True,
                 phosphated:bool=True,
                 irreversible:bool=False,
                 nick:str="nbI",
                 leak:float|None=None,
                 ):
        self.name=name
        self.input=input
        self.output=output
        self.sequence=sequence
        self.protected=protected
        self.irreversible=irreversible
        self.phosphated=phosphated
        self.concentration = concentration
        self.nick=nick
        self.leak=leak

    def print(self):
        print(f"Template: {self.name}")
        print(f"concentration: {self.concentration} nM")
        print(f"input signal: {self.input.name}")
        print(f"output signal: {self.output.name}")
        if self.sequence:
            print(f"sequence: {self.sequence}")
        print(f"protected: {self.protected}")
        print(f"phosphated: {self.phosphated}")
        print(f"nick: {self.nick}")
        print(f"irreversible: {self.irreversible}")
        print(f"leak: {self.leak}")


class Reporter:
    def __init__(self,
                 name:str,
                 input:Signal,
                 output:Signal|None=None,
                 concentration:float=0,
                 sequence:str|None=None,
                 fluorophore:str="?",
                 quencher:str="?",
                 reversible:bool=True
                 ):
        self.name=name
        self.input=input
        self.concentration = concentration
        self.sequence=sequence
        self.fluorophore=fluorophore
        self.reversible=reversible
        self.output=output
        self.quencher=quencher
        if output is None and reversible:
            raise ValueError("A reversible reporter should have an output signal.")

    def print(self):
        print(f"Reporter: {self.name}")
        print(f"concentration: {self.concentration} nM")
        print(f"Signal reported: {self.input.name}")
        print(f"sequence: {self.sequence}")
        print(f"fluorophore: {self.fluorophore}")
        print(f"quencher: {self.quencher}")
        print(f"reversible: {self.reversible}")



def reverse_complement(seq):
    """
    Returns the reverse complement of a DNA sequence.
    """
    complement = str.maketrans("ATCGatcg", "TAGCtagc")
    return seq.translate(complement)[::-1]