MPPT (Modified Physical Performance Test) – Video-Based Clinical Scoring Using MediaPipe

This project implements an automated Modified Physical Performance Test (MPPT) system using computer vision and pose estimation. Each MPPT task is performed by recording a short video using a standard camera. The video is analyzed frame-by-frame using MediaPipe Pose (and Hands where required) to extract human body landmarks, detect task-specific events, compute performance metrics, and generate standardized clinical scores.

For each task, the pipeline follows the same structure:
• Video input (live or recorded)
• Landmark extraction using MediaPipe
• Baseline calibration and noise filtering
• Rule-based or finite-state-machine (FSM) event detection
• Task-specific metric computation (time, repetitions, stability, completion)
• Conversion to MPPT-style ordinal score (0–4)
• Aggregation of scores for clinical interpretation

The system is designed to work with commodity cameras, requires no wearable sensors, and is robust to real-world recording conditions.

------------------------------------------------------------
Task 1: Standing Balance (Feet Together, Semi-Tandem, Full Tandem)

This task evaluates static balance and postural control. The subject performs three balance stances sequentially: feet together, semi-tandem, and full tandem.

The implementation tracks foot landmarks (heels and toes) using MediaPipe Pose and evaluates geometric alignment and spacing conditions specific to each stance. A valid posture must be maintained continuously for a fixed duration (typically 10 seconds). Timing is automatically started once a valid stance is detected and stopped when instability or misalignment occurs.

Each stance contributes to the final balance score based on the longest stable duration achieved. The task produces an overall balance score reflecting postural stability.

------------------------------------------------------------
Task 2: 50-Foot Walk Test

This task measures gait speed and functional mobility.

The system detects the start of walking using initial lower-body motion and tracks elapsed time while the subject walks a predefined distance. Distance completion is estimated using pose-based movement proxies, and stopping conditions are triggered when the target distance range is reached.

Walking speed is computed from distance and time, and the final score is assigned using MPPT timing thresholds. This task serves as a strong indicator of mobility limitations and fall risk.

------------------------------------------------------------
Task 3: Chair Rise (5x Sit-to-Stand)

This task evaluates lower-limb strength, endurance, and functional independence.

A finite-state machine detects sit-to-stand and stand-to-sit transitions using hip and shoulder vertical displacement relative to a seated baseline. The system counts five complete cycles and measures total completion time.

An additional compliance check verifies whether arms remain folded across the chest during repetitions. Timing violations and posture deviations are logged. The final score is based on total time to complete all five repetitions.

------------------------------------------------------------
Task 4: 360-Degree Turn

This task assesses dynamic balance, turning stability, and coordination.

The implementation computes body orientation changes using pose landmarks and accumulates rotational angle over time. Timing begins once a meaningful rotation is detected and ends when a full 360-degree rotation is completed.

The system also detects pauses during turning and estimates movement stability using vertical fluctuation metrics from hip and leg landmarks. Performance is summarized using turn duration, pauses, and stability indicators.

------------------------------------------------------------
Task 5: Stairs – One Flight

This task measures functional mobility and lower-limb coordination during stair ascent.

The system detects stair reference lines using edge detection and tracks ankle and hip movement to identify ascent start and completion. Timing begins when upward motion is confirmed and ends when the feet cross the top stair reference.

The final score is derived from total ascent time using MPPT scoring bands.

------------------------------------------------------------
Task 6: Stairs – Four-Step Round Trip (4x)

This task evaluates endurance and repeated stair negotiation ability.

A multi-phase finite-state machine tracks four sequential phases: ascent, descent, ascent, and descent. Each phase is validated using foot position relative to detected stair reference lines.

The score reflects the number of successfully completed phases (0–4). This task captures fatigue effects and sustained lower-limb performance.

------------------------------------------------------------
Task 7: Put On and Remove a Jacket

This task assesses upper-limb coordination and functional activities of daily living (ADLs).

The system monitors color and appearance changes around shoulder and forearm regions derived from pose landmarks. Baseline appearance is established before movement.

State transitions are detected when the jacket is put on and later removed using stabilized color differences over time. Timing is measured from task start to jacket removal completion, and the final score is assigned based on total task duration.

------------------------------------------------------------
Task 8: Pick Up a Nickel

This task measures trunk flexibility, balance, and recovery ability.

The system detects the start of movement after initial stabilization. The pick-up event is identified when the wrist landmark reaches near ankle height. The recovery phase is confirmed using hip and knee extension angles to ensure the subject returns to an upright posture.

Total time from movement start to full recovery determines the task score.

------------------------------------------------------------
Task 9: Lift a Book

This task evaluates upper-extremity functional strength and coordination.

Pose landmarks from the wrist, elbow, shoulder, and torso are tracked to detect the initiation of the lift, completion of the lifting motion, and controlled return. Timing and movement continuity are analyzed to identify pauses or instability.

The task score is based on completion time and smoothness of the lift, reflecting upper-limb functional capability.

------------------------------------------------------------
Final Scoring and Clinical Interpretation

Each task produces a standardized MPPT score (0–4). Individual task scores can be:
• Summed into a total MPPT score
• Grouped into functional domains (balance, gait, strength, ADLs)
• Mapped to clinical descriptors such as normal performance, mild impairment, moderate impairment, or severe impairment

This framework enables objective, repeatable, video-based physical performance assessment suitable for clinical screening, remote monitoring, and research applications.
