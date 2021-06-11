import pysynth as ps #wav 파일 생성
from pyknon.genmidi import Midi #midi 파일 생성
from pyknon.music import NoteSeq, Note, Rest
from src.MarkovMusic import MusicMatrix
from pprint import pprint

def make_midi(midi_path, notes, bpm=120):
    note_names = 'c c# d d# e f f# g g# a a# b'.split()

    result = NoteSeq()
    for n in notes:
        duration = 1. / n[1]

        if n[0].lower() == 'r':
            result.append(Rest(dur=duration))
        else:
            pitch = n[0][:-1]
            octave = int(n[0][-1]) + 1
            pitch_number = note_names.index(pitch.lower())

            result.append(Note(pitch_number, octave=octave, dur=duration))

    midi = Midi(number_tracks=1, tempo=bpm)
    midi.seq_notes(result, track=0)
    midi.write(midi_path)

#Row Row Row Your Boat

song = [['c4', 4], ['c4', 4], ['c4', 4], ['d4', 8], ['e4', 4], ['e4', 4], ['d4', 8], ['e4', 4], ['f4', 8], ['g4', 2], ['c4', 8], ['c4', 8], ['c4', 8], ['g4', 8], ['g4', 8], ['g4', 8], ['e4', 8], ['e4', 8], ['e4', 8], ['c4', 8], ['c4', 8], ['c4', 8], ['g4', 4], ['f4', 8], ['e4', 4], ['d4', 8], ['c4', 2]]

ps.make_wav(song, fn='examples/test.wav')