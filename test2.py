from diagrams import Diagram, Cluster, Node
from diagrams.custom import Custom

with Diagram("My Control System", show=True):
    input_signal = Custom("Input", "./path_to_input_icon.png")

    with Cluster("System"):
        subsystem1 = Custom("Subsystem 1", "./path_to_subsystem_icon.png")
        controller = Custom("Controller", "./path_to_controller_icon.png")
        subsystem2 = Custom("Subsystem 2", "./path_to_subsystem_icon.png")

    output_signal = Custom("Output", "./path_to_output_icon.png")

    input_signal >> subsystem1 >> controller >> subsystem2 >> output_signal
    output_signal - controller  # This line creates a feedback loop
