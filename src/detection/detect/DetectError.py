class NotEnoughDetectError(Exception):
  def __init__(self, msg):
    super().__init__(msg)
  pass