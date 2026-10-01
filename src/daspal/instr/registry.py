import daspal.instr.optodas as optodas
import daspal.instr.dasproc as dasproc
import daspal.instr.dxs as dxs
import daspal.instr.apsensing as apsensing

INSTRUMENTS = {
    "optodas": optodas.optodasInstr,
    "dxs": dxs.dxsInstr,
    "apsensing": apsensing.apsensingInstr,
    "dasproc": dasproc.dasprocInstr
    
}

def get_instrument(name):
    if name is None:
        raise ValueError(
            f"`instr` needs to be defined as: {', '.join(INSTRUMENTS.keys())}\n"
            f"if data formatted to daspal format use 'instr=dasproc'"
        )
    try:
        return INSTRUMENTS[name.lower()]()
    except KeyError:
        raise ValueError(f"Unknown instrument: {name}")