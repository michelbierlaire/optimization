"""Implementation of the school bus routing and scheduling problem inspired by Spada et al. (2005)

Classes for the data of the problem

Spada M., Bierlaire M., Liebling T. (2005). Decision-aid methodology
for the school bus routing and scheduling problem. Transportation Science 39 (4):477-490
https://dx.doi.org/10.1287/trsc.1040.0096

Michel Bierlaire
Mon May 26 2025, 10:38:44
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Origin:
    name: str

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, Origin):
            return NotImplemented
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


@dataclass
class Destination:
    name: str
    arrival_time: datetime

    def __str__(self):
        return f'{self.name} [{self.arrival_time}]'

    def __eq__(self, other):
        if not isinstance(other, Destination):
            return NotImplemented
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


@dataclass
class Bus:
    name: str
    capacity: float

    def __eq__(self, other):
        if not isinstance(other, Bus):
            return NotImplemented
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


@dataclass
class SchoolBusProblem:
    schools: set[Destination]
    origins: set[Origin]
    travel_times_in_minutes: dict[tuple[str, str], float]
    od_table: dict[tuple[str, str], float]
    buses: set[Bus]

    def is_location_known(self, location_name: str) -> bool:
        school = self.get_school(school_name=location_name)
        if school is not None:
            return True
        destination = self.get_origin(origin_name=location_name)
        return destination is not None

    def get_school(self, school_name: str) -> Destination | None:
        try:
            the_school = next(
                destination
                for destination in self.schools
                if destination.name == school_name
            )
        except StopIteration:
            return None
        return the_school

    def get_origin(self, origin_name: str) -> Origin | None:
        try:
            the_origin = next(
                origin for origin in self.origins if origin_name == origin.name
            )
        except StopIteration:
            return None
        return the_origin

    def get_travel_time(self, from_location: str, to_location: str) -> float:
        """Returns the travel time in minutes between two locations

        :param from_location: name of the origin location
        :param to_location: name of the destination location
        :return: travel time in minutes

        :raise ValueError: if the travel time is not available or one of the two locations is not known.
        """
        if not self.is_location_known(location_name=from_location):
            raise ValueError(f'Location {from_location} is unknown')
        if not self.is_location_known(location_name=to_location):
            raise ValueError(f'Location {to_location} is unknown')
        if from_location == to_location:
            return 0.0

        travel_time = self.travel_times_in_minutes.get((from_location, to_location))
        if travel_time is not None:
            return travel_time
        travel_time = self.travel_times_in_minutes.get((to_location, from_location))
        if travel_time is None:
            raise ValueError(
                f'No travel time information available for the pair {from_location}->{to_location}'
            )
        return travel_time

    def get_set_of_origins(self) -> set[str]:
        return {origin.name for origin in self.origins}

    def get_set_of_destinations(self):
        return {destination.name for destination in self.schools}

    def get_list_of_nodes(self) -> list[str]:
        """The list of nodes is organized so that the origins come first and the destinations after"""
        origins = self.get_set_of_origins()
        destinations = self.get_set_of_destinations()
        destinations_not_origins = destinations - origins
        return list(origins) + list(destinations_not_origins)
