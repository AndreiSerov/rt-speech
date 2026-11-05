# Live Call Copilot

A local tool that listens to both sides of a call, turns the speech into speaker-tagged text, and suggests what to say next.

## Language

**Stream**:
One of the two audio sources in a call: `system` (what the computer plays) or `mic` (what the microphone hears).
_Avoid_: Channel, track

**Speaker**:
The role a Stream's speech is attributed to: Client for `system`, Me for `mic`.
_Avoid_: Participant, person, voice

**Client**:
The Speaker for everything on the `system` Stream, including several remote participants and any non-call audio playing on the computer.
_Avoid_: Customer, remote, other side

**Me**:
The Speaker for everything on the `mic` Stream.
_Avoid_: User, agent, local

**Segment**:
One continuous stretch of speech cut from a single Stream, with its start and end times.
_Avoid_: Utterance, chunk, phrase

**Turn**:
Consecutive Segments from the same Speaker, closed by a change of Speaker or a long enough pause.
_Avoid_: Message, line, replica

**Suggestion**:
A short thing Me could say next, produced from recent Turns when a Client Turn closes.
_Avoid_: Hint, reply, prompt
