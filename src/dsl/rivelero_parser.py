from dataclasses import dataclass
from pathlib import Path

from lark import Lark, Transformer, UnexpectedInput


@dataclass(frozen=True) #once a rivelero command object is made, it is immutable.
class RiveleroCommand:
    name: str


class RiveleroTransformer(Transformer): #create a parse tree from the user text.
    def disphantom(self, _children): #if disphantom is called:
        return RiveleroCommand(name="DISPHANTOM")


grammar_path = Path(__file__).with_name("rivelero.lark") #look for a file named rivelero.lark in the same folder as this one.
grammar_text = grammar_path.read_text() #read it in.

parser = Lark(
    grammar_text,
    parser="lalr", #type is LALR.
    transformer=RiveleroTransformer(), #use transformer to turn parse tree into Rivelero command object.
) #create parser.


def parse_rivelero_command(command_text: str) -> RiveleroCommand: #for the use of GUI.py
    try:
        return parser.parse(command_text.strip())
    except UnexpectedInput as error:
        raise ValueError("Unknown Rivelero command. Try: Disphantom") from error