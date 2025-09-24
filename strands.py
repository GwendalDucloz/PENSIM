class Signal:
    def __init__(self,
                 name:str, 
                 concentration:float = 0, 
                 sequence:str = None, 
                 index=None, 
                 IsDrained:bool =False,
                 protected:bool=False
                 ):
        self.name=name
        self.sequence=sequence
        self.concentration=concentration
        self.IsDrained=IsDrained
        self.index=index
        self.protected=protected

    def get_index(self, option="alone"):
        """ Return the index of the signal"""
        if option=="alone":
            return self.index
        if option=="drained":
            if self.IsDrained :
                return self.index +1
            else:
                raise ValueError("""get_index called for "drained" but the signal is not drained""")
        else:
            return None

    def get_concentration(self,y,option="alone"):
        """Return the concentration of the signal"""
        if self.get_index(option)!= None:
            return y[self.get_index(option)]
        else:
            raise ValueError("Option not recognized")

class Drain:
    def __init__(self,
                 name:str,
                 input:Signal,
                 concentration:float=0,
                 index:int|None=None,
                 sequence:str|None=None,
                 protected:bool=False):
        self.name=name
        self.input=input
        self.index=index
        self.concentration = concentration
        self.sequence=sequence
        self.protected=protected

    def get_index(self, option="alone"):
        """ Return the index of the template"""
        if option=="alone":
            return self.index
        elif option=="in":
            return self.index + 1
        elif option=="ext":
            return self.index+ 2
        else:
            return None
        
    def get_concentration(self,y,option="alone"):
        """Return the concentration of the template"""
        if self.get_index(option):
            return y[self.get_index(option)]
        elif option=="total":
            return y[self.get_index("alone")] + y[self.get_index("in")] + y[self.get_index("out")]
        else:
            raise ValueError("Option not recognized")



class Template:
    def __init__(self, 
                 name:str, 
                 input:Signal, 
                 output:Signal|Drain, 
                 concentration:float = 0, 
                 sequence:str = None, 
                 index=None, 
                 protected:bool=True,
                 truncated:bool=False,
                 loading:bool=False,
                 nick:str="nbI"):
        self.name=name
        self.input=input
        self.output=output
        self.sequence=sequence
        self.protected=protected
        self.loading=loading
        self.index=index
        self.truncated=truncated
        self.concentration = concentration
        self.nick=nick            

    def get_index(self, option="alone"):
        """ Return the index of the template"""
        if option=="alone":
            return self.index
        elif option=="in":
            return self.index + 1
        elif option=="out":
            return self.index+ 2
        elif option=="both":
            return self.index+3
        elif option=="ext":
            return self.index+4
        elif option=='load':
            if self.loading:
                return self.index+5
            else:
                raise ValueError("""get_index called for "load" but the template is not loadd""")
        elif option=="in_drained":
            if self.input.IsDrained:
                if self.loading:
                    return self.index+6
                else:
                    return self.index+5
            else:
                raise ValueError("""get_index called for "in_drained" but the input is not drained""")
        else:
            return None
        
    def get_concentration(self,y,option="alone"):
        """Return the concentration of the template"""
        if self.get_index(option):
            return y[self.get_index(option)]
        elif option=="total":
            tot= y[self.get_index("alone")] + y[self.get_index("in")] + y[self.get_index("out")] + y[self.get_index("both")] + y[self.get_index("ext")]
            if self.loading:
                tot+= y[self.get_index("load")]
            return tot
        else:
            raise ValueError("Option not recognized")