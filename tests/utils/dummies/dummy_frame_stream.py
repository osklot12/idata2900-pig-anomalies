class DummyFrameStream:
    """A fake FrameStream that returns predefined frames."""
    def __init__(self, video_bytes):
        self.frames = [b"frame1", b"frame2", b"frame3"]
        self.index = 0

    def read(self):
        """Returns the next frame, or None if exhausted."""
        frame = b""
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
        return frame