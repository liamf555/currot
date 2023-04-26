# Do this import to ensure that the Gym environments get registered properly
import deep_sprl.environments
from .point_mass_2d_experiment import PointMass2DExperiment
from .mxs_box2d_experiment import MXSBox2DExperiment
from .abstract_experiment import CurriculumType, Learner

__all__ = ['CurriculumType', 'PointMass2DExperiment', 'MXSBox2DExperiment' 'Learner']
