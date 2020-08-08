# https://docs.bons.ai/references/inkling2-reference.html
inkling "2.0"
using Number
using Math


type SimState {
    position: number,
    velocity: number,
    angle: number,
    angular_velocity: number,
    gym_terminal: Number.Bool,
    sparse_reward: number<0,1,>
}

type ObservationState {
    position: number,
    velocity: number,
    angle: number,
    angular_velocity: number,
}

function Reward(state: SimState) {

    return state.sparse_reward

}

function Terminal(state: SimState) {

    return state.gym_terminal
}

const left: Number.Int8 = 0
const right: Number.Int8 = 1

type Action {
    command: Number.Int8<left, right,>
}

type SimConfig {
    # allowable variables for configuration in "scenarios"
    # config -> star.brain_to_sim_config -> simulator_integration.simulator_reset
    # make sure these are properly defined in both methods
    masspole: number,
    length: number
}

graph (input: ObservationState): Action {
    concept BalancePole(input): Action {
        curriculum {
            source simulator (Action: Action, Config: SimConfig): SimState {
            }
            reward Reward
            terminal Terminal
            lesson `Default` {
                scenario {
                    masspole: 0.1,
                    length: 0.5
                }
            }
            lesson `Randomize Length` {
                scenario {
                    masspole: 0.1,
                    length: number<0.1 .. 1>
                }
            }
            lesson `Randomize Length and Mass` {
                scenario {
                    masspole: number<0.1 .. 0.5>,
                    length: number<0.1 .. 1>
                }
            }
        }
    }
}