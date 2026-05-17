# ADI heat diffusion solver for agar gel phantom
CXX = g++
CXXFLAGS = -O2 -fopenmp -Wall -std=c++11
SRC_DIR = src
BUILD_DIR = build
TARGET = $(BUILD_DIR)/heat_diffusion_agar

SOURCES = $(SRC_DIR)/main.cpp \
          $(SRC_DIR)/material.cpp \
          $(SRC_DIR)/grid.cpp \
          $(SRC_DIR)/adi.cpp \
          $(SRC_DIR)/io.cpp

OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))

.PHONY: all clean run

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@

run: $(TARGET)
	cd $(BUILD_DIR) && ../$(TARGET)

clean:
	rm -rf $(BUILD_DIR)
