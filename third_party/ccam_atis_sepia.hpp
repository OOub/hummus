#pragma once

#include "sepia.hpp"
#include <array>
#include <libusb-1.0/libusb.h>

/// ccam_atis_sepia specialises sepia for the CCam ATIS.
/// In order to use this header, an application must link to the dynamic library usb-1.0.
namespace ccam_atis_sepia {

    /// camera represents a CCam ATIS.
    class camera {
        public:
        /// available_serials returns the connected CCam ATIS cameras' serials.
        static std::vector<uint16_t> available_serials() {
            std::vector<uint16_t> serials;
            libusb_context* context;
            check_usb_error(libusb_init(&context), "initializing the USB context");
            libusb_device** devices;
            const auto count = libusb_get_device_list(context, &devices);
            for (ssize_t index = 0; index < count; ++index) {
                libusb_device_descriptor descriptor;
                if (libusb_get_device_descriptor(devices[index], &descriptor) == 0) {
                    if (descriptor.idVendor == 1204 && descriptor.idProduct == 244) {
                        libusb_device_handle* handle;
                        check_usb_error(libusb_open(devices[index], &handle), "opening the device");
                        if (libusb_claim_interface(handle, 0) == 0) {
                            auto data = std::array<uint8_t, 8>{};
                            check_usb_error(
                                libusb_control_transfer(
                                    handle, 192, 85, 32, 0, data.data(), static_cast<uint16_t>(data.size()), 0),
                                "sending a control packet");
                            libusb_release_interface(handle, 0);
                            serials.push_back((static_cast<uint16_t>(data[6]) << 8) | static_cast<uint16_t>(data[7]));
                        }
                        libusb_close(handle);
                    }
                }
            }
            libusb_free_device_list(devices, 1);
            libusb_exit(context);
            return serials;
        }

        /// default_parameter returns the default parameter used by the CCam ATIS.
        static std::unique_ptr<sepia::parameter> default_parameter() {
            return sepia::make_unique<sepia::object_parameter>(
                "change_detection",
                sepia::make_unique<sepia::object_parameter>(
                    "reset_switch_bulk_potential",
                    sepia::make_unique<sepia::char_parameter>(207),
                    "photoreceptor_feedback",
                    sepia::make_unique<sepia::char_parameter>(243),
                    "refractory_period",
                    sepia::make_unique<sepia::char_parameter>(216),
                    "follower",
                    sepia::make_unique<sepia::char_parameter>(239),
                    "event_source_amplifier",
                    sepia::make_unique<sepia::char_parameter>(42),
                    "on_event_threshold",
                    sepia::make_unique<sepia::char_parameter>(51),
                    "off_event_threshold",
                    sepia::make_unique<sepia::char_parameter>(39),
                    "off_event_inverter",
                    sepia::make_unique<sepia::char_parameter>(61),
                    "cascode_photoreceptor_feedback",
                    sepia::make_unique<sepia::char_parameter>(154)),
                "exposure_measurement",
                sepia::make_unique<sepia::object_parameter>(
                    "comparator_tail",
                    sepia::make_unique<sepia::char_parameter>(54),
                    "comparator_hysteresis",
                    sepia::make_unique<sepia::char_parameter>(47),
                    "comparator_output_stage",
                    sepia::make_unique<sepia::char_parameter>(57),
                    "upper_threshold",
                    sepia::make_unique<sepia::char_parameter>(243),
                    "lower_threshold",
                    sepia::make_unique<sepia::char_parameter>(235)),
                "pullup",
                sepia::make_unique<sepia::object_parameter>(
                    "exposure_measurement_abscissa_request",
                    sepia::make_unique<sepia::char_parameter>(131),
                    "exposure_measurement_ordinate_request",
                    sepia::make_unique<sepia::char_parameter>(155),
                    "change_detection_abscissa_request",
                    sepia::make_unique<sepia::char_parameter>(151),
                    "change_detection_ordinate_request",
                    sepia::make_unique<sepia::char_parameter>(117),
                    "abscissa_acknoledge",
                    sepia::make_unique<sepia::char_parameter>(162),
                    "abscissa_encoder",
                    sepia::make_unique<sepia::char_parameter>(162),
                    "ordinate_encoder",
                    sepia::make_unique<sepia::char_parameter>(120)),
                "control",
                sepia::make_unique<sepia::object_parameter>(
                    "exposure_measurement_timeout",
                    sepia::make_unique<sepia::char_parameter>(49),
                    "sequential_exposure_measurement_timeout",
                    sepia::make_unique<sepia::char_parameter>(45),
                    "abscissa_acknoledge_timeout",
                    sepia::make_unique<sepia::char_parameter>(56),
                    "latch_cell_scan_pulldown",
                    sepia::make_unique<sepia::char_parameter>(134),
                    "abscissa_request_pulldown",
                    sepia::make_unique<sepia::char_parameter>(87)));
        }

        /// width returns the sensor width.
        static constexpr uint16_t width() {
            return 304;
        }

        /// height returns the sensor height.
        static constexpr uint16_t height() {
            return 240;
        }

        /// configuration contains the settings for the digital-to-analog converters on the FPGA.
        static std::
            unordered_map<std::string, std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>>>
            configuration() {
            return {
                {"change_detection",
                 {
                     {"reset_switch_bulk_potential", {{"address", 0x02}, {"tension", 0x5900}}},
                     {"photoreceptor_feedback", {{"address", 0x03}, {"tension", 0x5900}}},
                     {"refractory_period", {{"address", 0x04}, {"tension", 0x5900}}},
                     {"follower", {{"address", 0x05}, {"tension", 0x5900}}},
                     {"event_source_amplifier", {{"address", 0x06}, {"tension", 0x7900}}},
                     {"on_event_threshold", {{"address", 0x07}, {"tension", 0x7900}}},
                     {"off_event_threshold", {{"address", 0x08}, {"tension", 0x7900}}},
                     {"off_event_inverter", {{"address", 0x09}, {"tension", 0x7900}}},
                     {"cascode_photoreceptor_feedback", {{"address", 0x0a}, {"tension", 0x7900}}},
                 }},
                {"exposure_measurement",
                 {
                     {"comparator_tail", {{"address", 0x0b}, {"tension", 0x7900}}},
                     {"comparator_hysteresis", {{"address", 0x0c}, {"tension", 0x7900}}},
                     {"comparator_output_stage", {{"address", 0x0d}, {"tension", 0x7900}}},
                     {"upper_threshold", {{"address", 0x0e}, {"tension", 0x5900}}},
                     {"lower_threshold", {{"address", 0x0f}, {"tension", 0x5900}}},
                 }},
                {"pullup",
                 {
                     {"exposure_measurement_abscissa_request", {{"address", 0x10}, {"tension", 0x5900}}},
                     {"exposure_measurement_ordinate_request", {{"address", 0x11}, {"tension", 0x5900}}},
                     {"change_detection_abscissa_request", {{"address", 0x12}, {"tension", 0x5900}}},
                     {"change_detection_ordinate_request", {{"address", 0x13}, {"tension", 0x7900}}},
                     {"abscissa_acknoledge", {{"address", 0x14}, {"tension", 0x5900}}},
                     {"abscissa_encoder", {{"address", 0x15}, {"tension", 0x5900}}},
                     {"ordinate_encoder", {{"address", 0x16}, {"tension", 0x7900}}},
                 }},
                {"control",
                 {
                     {"exposure_measurement_timeout", {{"address", 0x17}, {"tension", 0x7900}}},
                     {"sequential_exposure_measurement_timeout", {{"address", 0x18}, {"tension", 0x7900}}},
                     {"abscissa_acknoledge_timeout", {{"address", 0x19}, {"tension", 0x7900}}},
                     {"latch_cell_scan_pulldown", {{"address", 0x1a}, {"tension", 0x5900}}},
                     {"abscissa_request_pulldown", {{"address", 0x1b}, {"tension", 0x7900}}},
                 }},
                {"static",
                 {
                     {"reset_t", {{"address", 0x00}, {"tension", 0x5900}, {"value", 0}}},
                     {"test_event", {{"address", 0x01}, {"tension", 0x7900}, {"value", 0}}},
                     {"reset_photodiodes", {{"address", 0x1c}, {"tension", 0x00}, {"value", 3}}},
                 }},
            };
        }

        camera() = default;
        camera(const camera&) = delete;
        camera(camera&&) = default;
        camera& operator=(const camera&) = delete;
        camera& operator=(camera&&) = default;
        virtual ~camera() {}

        /// trigger sends a trigger signal to the camera.
        /// with default settings, this signal will trigger a change detection on every pixel.
        virtual void trigger() = 0;

        protected:
        /// check_usb_error throws if the given value is not zero.
        static void check_usb_error(int error, const std::string& message) {
            if (error < 0) {
                throw std::logic_error(message + " failed: " + libusb_strerror(static_cast<libusb_error>(error)));
            }
        }

        /// send_command sends a setup command to the given camera.
        static void send_command(
            libusb_device_handle* handle,
            uint16_t w_value,
            std::array<uint8_t, 4> data,
            const std::string& message) {
            check_usb_error(
                libusb_control_transfer(handle, 64, 86, w_value, 0, data.data(), static_cast<uint16_t>(data.size()), 0),
                message);
        }
    };

    /// specialized_camera represents a template-specialized CCam ATIS.
    template <typename HandleEvent, typename HandleException>
    class specialized_camera : public camera,
                               public sepia::specialized_camera<sepia::atis_event, HandleEvent, HandleException> {
        public:
        specialized_camera<HandleEvent, HandleException>(
            HandleEvent handle_event,
            HandleException handle_exception,
            std::unique_ptr<sepia::unvalidated_parameter> unvalidated_parameter,
            std::size_t fifo_size,
            uint16_t serial,
            std::chrono::milliseconds sleep_duration) :
            sepia::specialized_camera<sepia::atis_event, HandleEvent, HandleException>(
                std::forward<HandleEvent>(handle_event),
                std::forward<HandleException>(handle_exception),
                fifo_size,
                sleep_duration),
            _parameter(default_parameter()),
            _acquisition_running(true) {
            _parameter->parse_or_load(std::move(unvalidated_parameter));

            // initialize the context
            check_usb_error(libusb_init(&_context), "initializing the USB context");

            // find requested / available devices
            {
                auto device_found = false;
                libusb_device** devices;
                const auto count = libusb_get_device_list(_context, &devices);
                for (ssize_t index = 0; index < count; ++index) {
                    libusb_device_descriptor descriptor;
                    if (libusb_get_device_descriptor(devices[index], &descriptor) == 0) {
                        if (descriptor.idVendor == 1204 && descriptor.idProduct == 244) {
                            check_usb_error(libusb_open(devices[index], &_handle), "opening the device");
                            if (libusb_claim_interface(_handle, 0) == 0) {
                                if (serial == 0) {
                                    device_found = true;
                                    break;
                                } else {
                                    auto data = std::array<uint8_t, 8>{};
                                    check_usb_error(
                                        libusb_control_transfer(
                                            _handle,
                                            192,
                                            85,
                                            32,
                                            0,
                                            data.data(),
                                            static_cast<uint16_t>(data.size()),
                                            0),
                                        "sending a control packet");
                                    if ((serial & 0xff) == data[6] && ((serial & 0xff00) >> 8) == data[7]) {
                                        device_found = true;
                                        break;
                                    }
                                }
                                libusb_release_interface(_handle, 0);
                            }
                            libusb_close(_handle);
                        }
                    }
                }
                libusb_free_device_list(devices, 1);
                if (!device_found) {
                    libusb_exit(_context);
                    throw sepia::no_device_connected("CCam ATIS");
                }
            }

            // allocate a transfer
            _transfer = libusb_alloc_transfer(0);

            // send setup commands to the camera
            check_usb_error(libusb_reset_device(_handle), "resetting the device");
            send_command(_handle, 0x01a, {0, 0, 0x00, 0x01}, "setting the role");
            send_command(_handle, 0x41a, {0, 0, 0x00, 0x02}, "setting the role");
            {
                auto data = std::vector<uint8_t>{};
                for (const auto& category_pair : configuration()) {
                    for (const auto& setting_pair : category_pair.second) {
                        {
                            const auto value =
                                (category_pair.first == "static" ? setting_pair.second.at("value") :
                                                                   static_cast<uint32_t>(_parameter->get_number(
                                                                       {category_pair.first, setting_pair.first})));
                            data.push_back(static_cast<uint8_t>(value >> 24));
                            data.push_back(static_cast<uint8_t>(value >> 16));
                            data.push_back(static_cast<uint8_t>(value >> 8));
                            data.push_back(static_cast<uint8_t>(value & 0xff));
                        }
                        {
                            const auto tension = setting_pair.second.at("tension");
                            data.push_back(static_cast<uint8_t>(tension >> 24));
                            data.push_back(static_cast<uint8_t>(tension >> 16));
                            data.push_back(static_cast<uint8_t>(tension >> 8));
                            data.push_back(static_cast<uint8_t>(tension & 0xff));
                        }
                        {
                            const auto address = setting_pair.second.at("address");
                            data.push_back(static_cast<uint8_t>(address >> 24));
                            data.push_back(static_cast<uint8_t>(address >> 16));
                            data.push_back(static_cast<uint8_t>(address >> 8));
                            data.push_back(static_cast<uint8_t>(address & 0xff));
                        }
                    }
                }
                check_usb_error(
                    libusb_control_transfer(_handle, 64, 97, 0, 0, data.data(), static_cast<uint16_t>(data.size()), 0),
                    "loading the biases");
                check_usb_error(
                    libusb_control_transfer(_handle, 64, 98, 0, 0, data.data(), static_cast<uint16_t>(data.size()), 0),
                    "loading the biases");
            }
            send_command(_handle, 0x00a, {0, 0, 0x00, 0x40}, "flush the biases");
            send_command(_handle, 0x40a, {0, 0, 0x00, 0x40}, "flush the biases");
            send_command(_handle, 0x008, {0, 0, 0x03, 0x2c}, "set the mode");
            send_command(_handle, 0x408, {0, 0, 0x03, 0x2c}, "set the mode");
            {
                auto data = std::array<uint8_t, 1024>{};
                int32_t transferred;
                libusb_bulk_transfer(_handle, 129, data.data(), data.size(), &transferred, 100);
            }
            send_command(_handle, 0x000, {0, 0, 0x0c, 0x81}, "start reading");
            send_command(_handle, 0x400, {0, 0, 0x0c, 0x81}, "start reading");

            // start the reading loop
            _acquisition_loop = std::thread([this, serial]() -> void {
                try {
                    auto data = std::vector<uint8_t>(1 << 17);
                    sepia::atis_event event;
                    uint64_t t_offset;
                    while (_acquisition_running.load(std::memory_order_relaxed)) {
                        int32_t transferred = 0;
                        const auto error = libusb_bulk_transfer(
                            _handle,
                            129,
                            data.data(),
                            static_cast<uint32_t>(data.size()),
                            &transferred,
                            static_cast<uint32_t>(this->_sleep_duration.count()));
                        if ((error == 0 || error == LIBUSB_ERROR_TIMEOUT) && transferred % 4 == 0) {
                            for (auto byte_iterator = data.begin();
                                 byte_iterator != std::next(data.begin(), transferred);
                                 std::advance(byte_iterator, 4)) {
                                if (*std::next(byte_iterator, 3) == 0x80) {
                                    t_offset = (static_cast<uint64_t>(*byte_iterator)
                                                | (static_cast<uint64_t>(*std::next(byte_iterator, 1)) << 8)
                                                | (static_cast<uint64_t>(*std::next(byte_iterator, 2)) << 16))
                                               * 0x800;
                                } else {
                                    event.x = (static_cast<uint16_t>(*std::next(byte_iterator, 2) & 0x1) << 8)
                                              + *std::next(byte_iterator, 1);
                                    event.y = 239 - *byte_iterator;
                                    event.t = t_offset
                                              + ((static_cast<uint64_t>((*std::next(byte_iterator, 3) & 0xf)) << 7)
                                                 | (*std::next(byte_iterator, 2) >> 1));
                                    event.polarity = ((*std::next(byte_iterator, 3) & 0b10000) >> 4) == 1;
                                    event.is_threshold_crossing = ((*std::next(byte_iterator, 3) & 0b100000) >> 5) == 1;
                                    if (!this->push(event)) {
                                        throw std::runtime_error("computer's FIFO overflow");
                                    }
                                }
                            }
                        } else {
                            throw sepia::device_disconnected("CCam ATIS");
                        }
                    }
                } catch (...) {
                    this->_handle_exception(std::current_exception());
                }
            });
        }
        specialized_camera(const specialized_camera&) = delete;
        specialized_camera(specialized_camera&&) = default;
        specialized_camera& operator=(const specialized_camera&) = delete;
        specialized_camera& operator=(specialized_camera&&) = default;
        virtual ~specialized_camera() {
            _acquisition_running.store(false, std::memory_order_relaxed);
            _acquisition_loop.join();
            libusb_release_interface(_handle, 0);
            libusb_free_transfer(_transfer);
            libusb_close(_handle);
            libusb_exit(_context);
        }
        virtual void trigger() override {
            // @TODO trigger the camera
        }

        protected:
        std::unique_ptr<sepia::parameter> _parameter;
        std::atomic_bool _acquisition_running;
        libusb_context* _context;
        libusb_device_handle* _handle;
        libusb_transfer* _transfer;
        std::thread _acquisition_loop;
    };

    /// make_camera creates a camera from functors.
    template <typename HandleEvent, typename HandleException>
    std::unique_ptr<specialized_camera<HandleEvent, HandleException>> make_camera(
        HandleEvent handle_event,
        HandleException handle_exception,
        std::unique_ptr<sepia::unvalidated_parameter> unvalidated_parameter =
            std::unique_ptr<sepia::unvalidated_parameter>(),
        std::size_t fifo_size = 1 << 24,
        uint16_t serial = 0,
        std::chrono::milliseconds sleep_duration = std::chrono::milliseconds(10)) {
        return sepia::make_unique<specialized_camera<HandleEvent, HandleException>>(
            std::forward<HandleEvent>(handle_event),
            std::forward<HandleException>(handle_exception),
            std::move(unvalidated_parameter),
            fifo_size,
            serial,
            sleep_duration);
    }
}
