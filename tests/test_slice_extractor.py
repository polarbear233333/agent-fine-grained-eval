from td_pipeline.trajectory_schema import Trajectory, TrajectoryStep
from td_pipeline.slice_extractor import extract_slices


def test_extract_basic_plan():
    rubric = {"planning_triggers": {"strong": ["plan:"], "revision": ["new plan"]}}
    traj = Trajectory(instance_id="x", steps=[
        TrajectoryStep(turn_id=0, thought="Plan:\n1. Reproduce\n2. Locate\n3. Fix\n4. Verify"),
        TrajectoryStep(turn_id=1, thought="Run tests", action={"command": "pytest"}, observation="failed"),
        TrajectoryStep(turn_id=2, thought="Verify", action={"command": "pytest"}, observation="passed"),
    ])
    slices = extract_slices(traj, rubric)
    assert len(slices) == 1
    assert slices[0].start_turn == 0
    assert "pytest" in slices[0].execution_text
