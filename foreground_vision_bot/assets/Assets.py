import json
import os
import shutil
from pathlib import Path

import cv2 as cv

mob_type_wind_path = str(Path(__file__).parent / "mob_types" / "wind.png")
mob_type_fire_path = str(Path(__file__).parent / "mob_types" / "fire.png")
mob_type_soil_path = str(Path(__file__).parent / "mob_types" / "soil.png")
mob_type_water_path = str(Path(__file__).parent / "mob_types" / "water.png")
mob_type_electricity_path = str(Path(__file__).parent / "mob_types" / "electricity.png")

mob_life_bar_path = str(Path(__file__).parent / "general" / "mob_life_bar.png")
user_target_bar_path = str(Path(__file__).parent / "general" / "user_target_bar.png")
inventory_perin_converter_path = str(
    Path(__file__).parent / "general" / "inventory_perin_converter.png"
)
inventory_icons_path = str(Path(__file__).parent / "general" / "inventory_icons.png")


class MobType:
    WIND = cv.imread(mob_type_wind_path, cv.IMREAD_GRAYSCALE)
    FIRE = cv.imread(mob_type_fire_path, cv.IMREAD_GRAYSCALE)
    SOIL = cv.imread(mob_type_soil_path, cv.IMREAD_GRAYSCALE)
    WATER = cv.imread(mob_type_water_path, cv.IMREAD_GRAYSCALE)
    ELECTRICITY = cv.imread(mob_type_electricity_path, cv.IMREAD_GRAYSCALE)


class MobInfo:
    @staticmethod
    def add_new_mob(
        name: str,
        map_name: str,
        image_path: str | None,
        height_offset: int,
        element: str,
        species_id: int | None = None,
    ) -> None:
        """
        Add new mob to json collection (mobs_list.json)
        """
        json_collection_path = str(Path(__file__).parent / "mobs_list.json")

        # The native reader does not need a name image. Keep optional image
        # support for the legacy CV preview/detector.
        if image_path:
            source_path = Path(image_path).expanduser()
            destination_path = Path(__file__).parent / "names" / f"{name}.png"

            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Legacy CV image does not exist: {source_path}"
                )

            destination_path.parent.mkdir(parents=True, exist_ok=True)

            # Updating an existing mob often points the optional image picker
            # at the image already stored in assets/names. shutil.copyfile
            # raises SameFileError in that case, even though no copy is needed.
            same_file = False
            if destination_path.exists():
                try:
                    same_file = source_path.samefile(destination_path)
                except OSError:
                    # Fall back to normalized absolute paths if the platform
                    # cannot perform an identity check for either path.
                    same_file = os.path.normcase(os.path.abspath(source_path)) == (
                        os.path.normcase(os.path.abspath(destination_path))
                    )

            if not same_file:
                shutil.copyfile(source_path, destination_path)

        current_mobs_list = MobInfo.get_all_mobs()
        current_mobs_list[name] = {
            "name": name,
            "element": element,
            "map_name": map_name,
            "height_offset": height_offset,
        }
        if species_id is not None:
            if isinstance(species_id, bool) or int(species_id) < 0:
                raise ValueError("species_id must be a non-negative integer")
            current_mobs_list[name]["species_id"] = int(species_id)

        with open(json_collection_path, "w+") as file:
            json.dump(current_mobs_list, file)

    @staticmethod
    def delete_mobs(name_list: list[str]) -> None:
        """
        Delete list of mobs from json collection (mobs_list.json)
        """
        json_collection_path = str(Path(__file__).parent / "mobs_list.json")
        current_mobs_list = MobInfo.get_all_mobs()
        new_mobs_list = {}

        # delete images for cv detection from asset folder
        for key in current_mobs_list:
            if key in name_list:
                image_path = Path(__file__).parent / "names" / f"{key}.png"
                if image_path.exists():
                    os.remove(str(image_path))
            else:
                new_mobs_list[key] = current_mobs_list[key]

        with open(json_collection_path, "w+") as file:
            json.dump(new_mobs_list, file)

    @staticmethod
    def get_all_mobs() -> dict[str, dict]:
        """
        Get a list of all mobs registered. Using a dump of mobs_list.json file

        :return: list of all mobs as dict (key: 'mob_name', val: params_dict)
        """
        json_collection_path = str(Path(__file__).parent / "mobs_list.json")

        # Check mobs_list.json
        if not os.path.isfile(json_collection_path):
            with open(json_collection_path, "w+") as file:
                json.dump({}, file)

        with open(json_collection_path) as file:
            mobs_list = json.load(file)
        return mobs_list


class GeneralAssets:
    MOB_LIFE_BAR = cv.imread(mob_life_bar_path, cv.IMREAD_GRAYSCALE)
    USER_TARGET_BAR = cv.imread(user_target_bar_path, cv.IMREAD_GRAYSCALE)
    INVENTORY_PERIN_CONVERTER = cv.imread(
        inventory_perin_converter_path, cv.IMREAD_GRAYSCALE
    )
    INVENTORY_ICONS = cv.imread(inventory_icons_path, cv.IMREAD_GRAYSCALE)
