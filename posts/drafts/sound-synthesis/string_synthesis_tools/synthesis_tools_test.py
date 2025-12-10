# Create a dataclass for the initial conditions of the waveguide


import numpy as np


# @dataclass
class WaveguideStates:
    positions: np.ndarray
    velocities: np.ndarray
    linear_density_kg_per_m: float  # kg/m
    tension_N: float  # Newtons
    length_m: float  # meters
    simulation_sampling_rate_hz: int  # Hz
    time_samples: int = 0

    @property
    def theoretical_wave_speed(self) -> float:
        return np.sqrt(self.tension_N / self.linear_density_kg_per_m)

    @property
    def dx(self) -> float:
        return self.length_m / (self.positions.shape[1] - 1)

    @property
    def c(self) -> float:
        return self.theoretical_wave_speed

    @property
    def dt(self) -> float:
        # return self.dx / self.c / 4  # CFL condition with safety factor
        return 1.0 / self.simulation_sampling_rate_hz

    def verify_cfl_condition(self) -> bool:
        cfl_number = (self.c * self.dt) / self.dx
        if cfl_number <= 1.0:
            raise ValueError(
                f"CFL condition not met: c*dt/dx = {cfl_number} > 1.0. c={self.c}, dt={self.dt}, dx={self.dx}"
            )
        return True

    @property
    def theoretical_fundamental_frequency(self) -> float:
        return self.c / (2 * self.length_m)

    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        time: float = 0.0,
        verify_cfl: bool = False,
    ):
        # If positions and velocities are 1D arrays, convert them to 2D column vectors
        if positions.ndim == 1:
            positions = positions[np.newaxis, :]
        if velocities.ndim == 1:
            velocities = velocities[np.newaxis, :]

        self.positions = positions
        self.velocities = velocities
        self.time = time

        if verify_cfl:
            self.verify_cfl_condition()
        if self.positions.shape != self.velocities.shape:
            raise ValueError(
                "Positions and velocities must have the same shape."
                f"Got {self.positions.shape} and {self.velocities.shape}."
            )

    @classmethod
    def initialize(cls, length: int, time: float) -> "WaveguideStates":
        positions = np.zeros((1, length), dtype=np.float64)
        velocities = np.zeros((1, length), dtype=np.float64)
        return cls(positions=positions, velocities=velocities, time=time)

    @classmethod
    def initial_pluck(
        cls, length: int, pluck_position_p: float, pluck_amplitude: float
    ) -> "WaveguideStates":
        positions = np.zeros(length)
        velocities = np.zeros(length)

        pluck_position = int(pluck_position_p * (length - 1))

        # Create a triangular pluck shape
        for i in range(length):
            if i < pluck_position:
                positions[i] = (pluck_amplitude / pluck_position) * i
            else:
                positions[i] = pluck_amplitude - (
                    pluck_amplitude / (length - pluck_position)
                ) * (i - pluck_position)
        return cls(positions=positions, velocities=velocities)

    @classmethod
    def initial_strike(
        cls,
        length: int,
        strike_position_p: float,
        strike_amplitude: float,
        strike_width: int,
    ) -> "WaveguideStates":
        positions = np.zeros(length)
        velocities = np.zeros(length)
        strike_position = int(strike_position_p * (length - 1))

        # Create a rectangular strike shape
        start = max(0, strike_position - strike_width // 2)
        end = min(length, strike_position + strike_width // 2)
        velocities[start:end] = strike_amplitude

        return cls(positions=positions, velocities=velocities)

    @classmethod
    def initial_standing_wave(
        cls, length: int, harmonic_amplitudes: dict[int, float] | None = None, **kwargs
    ) -> "WaveguideStates":
        positions = np.zeros(length)
        velocities = np.zeros(length)

        x = np.linspace(0, np.pi, length)

        if harmonic_amplitudes is not None:
            for harmonic, amplitude in harmonic_amplitudes.items():
                positions += amplitude * np.sin(harmonic * x)

        return cls(positions=positions, velocities=velocities, **kwargs)

    def plot(
        self,
        index: int = -1,
        height: int = 400,
        y_range: tuple[float, float] = (-0.06, 0.06),
    ) -> None:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(y=self.positions[index], mode="lines", name="Positions")
        )
        fig.add_trace(
            go.Scatter(y=self.velocities[index], mode="lines", name="Velocities")
        )
        fig.update_layout(
            title="Waveguide State",
            xaxis_title="Sample Index",
            yaxis_title="Amplitude",
            legend_title="Legend",
            height=height,
            yaxis=dict(range=y_range),
        )
        fig.show()

    def plot_animation(
        self,
        height: int = 400,
        y_range: tuple[float, float] = (-0.08, 0.08),
        interval: int = 100,
        skip_frames: int = 50,
        num_frames: int | None = 100,
    ) -> None:
        import plotly.graph_objects as go

        positions = self.positions[::skip_frames]
        velocities = self.velocities[::skip_frames]

        if num_frames:
            positions = positions[:num_frames]
            velocities = velocities[:num_frames]

        fig = go.Figure(
            data=[
                go.Scatter(y=positions[0], mode="lines", name="Positions"),
                go.Scatter(y=velocities[0], mode="lines", name="Velocities"),
            ],
            layout=go.Layout(
                title="Waveguide State Animation",
                xaxis_title="Sample Index",
                yaxis_title="Amplitude",
                legend_title="Legend",
                height=height,
                yaxis=dict(range=y_range),
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[
                                    None,
                                    {
                                        "frame": {"duration": interval, "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {
                                            "duration": 0,
                                            #    "easing": "cubic"
                                        },
                                    },
                                ],
                            ),
                            dict(
                                label="Pause",
                                method="animate",
                                args=[
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                            ),
                        ],
                    )
                ],
            ),
            frames=[
                go.Frame(
                    data=[
                        go.Scatter(y=positions[k], mode="lines"),
                        go.Scatter(y=velocities[k], mode="lines"),
                    ],
                    name=f"frame{k}",
                )
                for k in range(positions.shape[0])
            ],
        )

        fig.update_layout(
            sliders=[
                dict(
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [f"frame{k}"],
                                {
                                    "frame": {"duration": interval, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            label=f"{k}",
                        )
                        for k in range(positions.shape[0])
                    ],
                    transition={"duration": 0},
                    x=0,
                    y=0,
                    currentvalue=dict(
                        font=dict(size=12), prefix="Frame: ", visible=True
                    ),
                    len=1.0,
                )
            ]
        )

        fig.show()

    def get_audio(
        self,
        string_position: float = 0.43,
    ) -> np.ndarray:
        string_position_int = int(string_position * (self.positions.shape[1] - 1))

        audio = np.array(
            [position[string_position_int] for position in self.positions],
            dtype=np.float32,
        )

        return audio

    def calculate_fundamental_frequency(self, method: str = "zero_crossing") -> float:
        audio = self.get_audio(string_position=0.50)
        audio -= np.mean(audio)  # Remove DC offset

        if method == "zero_crossing":
            zero_crossings = np.where(np.diff(np.sign(audio)))[0]
            if len(zero_crossings) < 2:
                return 0.0  # Not enough zero crossings to calculate frequency
            periods = np.diff(zero_crossings)
            average_period = np.mean(periods)
            sampling_rate = 1.0 / self.dt
            # account for the fact that each period has two zero crossings
            average_period *= 2
            fundamental_frequency = sampling_rate / average_period
            return fundamental_frequency
        else:
            raise ValueError(f"Unknown method: {method}")

    def play_audio(
        self,
        sample_rate: int = 48000,
        duration: float | None = None,
        string_position: float = 0.43,
    ) -> None:
        import IPython.display as ipd
        from IPython.display import Audio
        from scipy.signal import resample_poly

        audio = self.get_audio(string_position=string_position)

        # Check that the sampling rate of the simulation is an integer multiple of the desired sample rate
        if self.simulation_sampling_rate_hz % sample_rate != 0:
            raise ValueError(
                f"Simulation sampling rate ({self.simulation_sampling_rate_hz} Hz) must be an integer multiple of the desired sample rate ({sample_rate} Hz)."
            )

        ratio = self.simulation_sampling_rate_hz // sample_rate
        print(
            f"Resampling audio by a factor of {ratio} from {self.simulation_sampling_rate_hz} Hz to {sample_rate} Hz."
        )

        audio = resample_poly(audio, up=1, down=ratio)

        ipd.display(ipd.Audio(audio, rate=sample_rate), autoplay=True)
        # audio_data = self.positions.flatten()
        return Audio(
            data=audio,
            rate=sample_rate,
            autoplay=False,
        )

    def fundamental_error(self, expected_frequency: float | None = None) -> float:
        calculated_freq = self.calculate_fundamental_frequency()

        expected_frequency = (
            expected_frequency or self.theoretical_fundamental_frequency
        )
        error = abs(calculated_freq - expected_frequency) / expected_frequency
        return error


# standing_wave_state.positions.shape
# standing_wave_state.plot()


def simulate_waveguide(
    initial_state: WaveguideStates,
    num_steps: int,
    c: float = 1.0,
    dx: float = 1.0,
    # dt: float = 0.1,
    verlet: bool = True,
) -> WaveguideStates:
    length = len(initial_state.positions[-1])
    position = initial_state.positions[-1].copy()
    velocity = initial_state.velocities[-1].copy()
    time = initial_state.time
    dt = initial_state.dt

    positions_to_append = []
    velocities_to_append = []

    for step in range(num_steps):
        new_position = position.copy()
        new_velocity = velocity.copy()

        for i in range(1, length - 1):
            # new_velocity[i] += (c**2 * dt / dx**2) * (
            #     position[i + 1] - 2 * position[i] + position[i - 1]
            # )
            c = initial_state.theoretical_wave_speed
            dx = initial_state.dx
            dt = initial_state.dt
            new_velocity[i] += (c**2 * dt / dx**2) * (
                position[i + 1] - 2 * position[i] + position[i - 1]
            )
        # Update position using velocity and acceleration (Verlet integration)
        new_position += new_velocity * dt
        if verlet:
            new_position += (
                0.5 * (new_velocity - velocity) * dt**2
            )  # Verlet term. Seems to add a LP filter

        position = new_position
        velocity = new_velocity
        time += dt

        # if step % 100 == 0:
        # print(f"Step {step}/{num_steps}")
        positions_to_append.append(position.copy())
        velocities_to_append.append(velocity.copy())

    new_wave_guide_states = initial_state.__class__(
        positions=np.vstack([initial_state.positions] + positions_to_append),
        velocities=np.vstack([initial_state.velocities] + velocities_to_append),
        time=time,
    )

    return new_wave_guide_states


# Example usage
# standing_wave_state.plot()

# # plucked_state = WaveguideStates.initial_pluck(
# #     length=N, pluck_position_p=0.3, pluck_amplitude=0.05
# # )
# # plucked_state.plot()


# # struck_state = WaveguideStates.initial_strike(
# #     length=N, strike_position_p=0.5, strike_amplitude=0.005, strike_width=10
# # )
# # struck_state.plot()


class _WaveguideStates(WaveguideStates):
    linear_density_kg_per_m: float = 0.0081 / 4  # kg/m
    tension_N: float = 186.650244  # Newtons
    length_m: float = 0.69  # meters
    simulation_sampling_rate_hz: int = 192000  # Hz


def test_waveguide_standing_wave_verlet():
    N = 20
    standing_wave_state = _WaveguideStates.initial_standing_wave(
        length=N,
        harmonic_amplitudes={
            1: 0.01,
        },
        verify_cfl=False,
    )

    output = simulate_waveguide(
        standing_wave_state, num_steps=192000 // 64, verlet=True
    )
    assert output.fundamental_error(expected_frequency=220) < 0.002


def test_waveguide_standing_wave_euler():
    N = 20
    standing_wave_state = _WaveguideStates.initial_standing_wave(
        length=N,
        harmonic_amplitudes={
            1: 0.01,
        },
        verify_cfl=False,
    )

    output = simulate_waveguide(
        standing_wave_state, num_steps=192000 // 64, verlet=False
    )
    assert output.fundamental_error(expected_frequency=220) < 0.002
