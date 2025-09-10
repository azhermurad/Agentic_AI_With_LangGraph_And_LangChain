from dataclasses import dataclass
from typing import Literal, TypedDict, Annotated

from pydantic import BaseModel

# different ways to define the state in the langGraph


# Distype


class State(TypedDict):
    messages: Annotated[list, "list of messages"]


# dataclass to define the state of the graph
@dataclass
class DataClassState:
    name: str
    mood: Literal["happy", "sad"]


dataclass = DataClassState(name="azher ali", mood="happyss")
print(dataclass)


# pydantic to define the state of the graph
# pydantic validate the type of the field as well during the runtime


class PydanticClass(BaseModel):
    name: str
    mood: Literal["happy", "mood"]



pydantic_object = PydanticClass(name="azher ali", mood="happy")