from dataclasses import dataclass, field
from squat_set import SquatSet

@dataclass
class Dataset:
    allSets: list[SquatSet] = field(default_factory=list)
    trainingSet: list[SquatSet] = field(default_factory=list)
    testingSet: list[SquatSet] = field(default_factory=list)
    #Path to both files
    trainingSetPath: str
    testingSetPath: str
    