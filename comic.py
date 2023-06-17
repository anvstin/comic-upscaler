from pathlib import Path

class Comic:
    outputPath = "./output"

    @staticmethod
    def fromOutputPath(path: Path) -> "Comic":
        return Comic(path.relative_to(Comic.outputPath))

    def __init__(self, path) -> None:
        self.path = Path(path)

    @property
    def titleName(self) -> str:
        # The parent directory of the comic is the title
        return self.path.parent.name

    @property
    def chapterName(self) -> str:
        return self.path.name

    @property
    def titlePath(self) -> Path:
        return self.path.parent

    @property
    def chapterPath(self) -> Path:
        return self.path

    @property
    def chapterNumber(self) -> int:
        # The chapter number is the last number in the chapter name
        numbers = [int(s) for s in self.chapterName.split() if s.isdigit()]
        return numbers[-1]

    def generateXMLData(self) -> str:
        # Generate the XML file for the comic
        return f"""<?xml version="1.0" encoding="UTF-8"?>
        <ComicInfo xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns="http://comicrack.cyolito.com/ComicInfo">
        <Series>{self.titleName}</Series>
        <Number>{self.chapterNumber}</Number>
        <Title>{self.chapterName}</Title>
        </ComicInfo>"""

    def generateOutputPath(self) -> Path:
        # Generate the output path for the comic
        return Path(self.outputPath).joinpath(self.titleName, self.chapterName)
