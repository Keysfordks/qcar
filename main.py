import time
from qvl.qlabs import QuanserInteractiveLabs as QLabs
from qvl.qcar import QLabsQCar

def main():
    qlabs = QLabs()
    qlabs.open("localhost")

    qcar = QLabsQCar(qlabs)
    qcar.spawn(location=[-13.563, -7.293, 0.005], rotation=[0, 0, -0.717], scale=[1, 1, 1])
    qcar.possess(qcar.CAMERA_CSI_FRONT)

    forward_speed = 1.0
    steering_angle = 0.0

    try:
        while True:
            qcar.set_velocity_and_request_state(
                forward=forward_speed,
                turn=steering_angle,
                headlights=True,
                leftTurnSignal=False,
                rightTurnSignal=False,
                brakeSignal=False,
                reverseSignal=False
            )
            print("Main: Driving forward...")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Main: Stopping...")

    qlabs.close()

if __name__ == "__main__":
    main()
