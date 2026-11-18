/**
 * @file Exception.h
 * @brief Custom exception class with enhanced error reporting for polymer FTS library.
 *
 * This header provides a custom exception class that extends std::runtime_error
 * with additional context information including file name, line number, and
 * function name where the exception was thrown. This facilitates debugging by
 * providing precise location information in error messages.
 *
 * @note Use the provided macros throw_with_line_number() and throw_without_line_number()
 *       instead of throwing Exception directly to automatically capture location info.
 *
 * **Rethrowing.** A catch block that rethrows what it caught with the caught message,
 *
 * @code
 * catch (std::exception& exc)
 * {
 *     release();
 *     throw_without_line_number(exc.what());
 * }
 * @endcode
 *
 * passes an Exception through unchanged, so a message names the place where the error
 * was found once; any other exception (std::out_of_range, std::bad_alloc, ...) becomes an
 * Exception at the catch block, which reaches Python as RuntimeError.
 *
 * @example
 * @code
 * // Throwing an exception with line number information
 * if (invalid_parameter) {
 *     throw_with_line_number("Invalid parameter: value must be positive");
 * }
 *
 * // Output: File: "src/common/MyClass.cpp", line: 42, function <myFunction>
 * //         Invalid parameter: value must be positive
 * @endcode
 */

#ifndef FTS_Exception_H_
#define FTS_Exception_H_

#include <exception>
#include <string>
#include <stdexcept>

/**
 * @class Exception
 * @brief Enhanced runtime exception with source location information.
 *
 * Extends std::runtime_error to include file name, line number, and function
 * name in the error message. This provides detailed context for debugging
 * when exceptions are caught and logged.
 *
 * @note Prefer using throw_with_line_number() or throw_without_line_number()
 *       macros which automatically capture __FILE__, __LINE__, and __func__.
 */
class Exception : public std::runtime_error {
public:
    /**
     * @brief Construct exception with full location information.
     *
     * @param arg   Error message describing the exception
     * @param file  Source file name (use __FILE__ macro)
     * @param line  Line number in source file (use __LINE__ macro)
     * @param func  Function name (use __func__ macro)
     *
     * @example
     * @code
     * throw Exception("Memory allocation failed", __FILE__, __LINE__, __func__);
     * @endcode
     */
    Exception(const std::string &arg, const char *file, int line, const char *func):
        std::runtime_error("File: \"" + std::string(file) + "\", line: " + std::to_string(line) + ", function <" + std::string(func) + ">\n\t" + arg){};

    /**
     * @brief Construct exception without line number.
     *
     * @param arg   Error message describing the exception
     * @param file  Source file name (use __FILE__ macro)
     * @param func  Function name (use __func__ macro)
     *
     * @example
     * @code
     * throw Exception("Configuration error", __FILE__, __func__);
     * @endcode
     */
    Exception(const std::string &arg, const char *file, const char *func):
        std::runtime_error("File: \"" + std::string(file) + "\", function <" + std::string(func) + ">\n\t" + arg){};

    /// Destructor (noexcept).
    ~Exception() noexcept {}
};

namespace exception_detail
{
/// True in a catch block that handles an Exception with this message: rethrowing it
/// with its own message adds nothing, since it names where it was thrown already.
inline bool handles_exception_with_message(const std::string& message)
{
    const std::exception_ptr handled = std::current_exception();
    if (!handled)
        return false;
    try
    {
        std::rethrow_exception(handled);
    }
    catch (const Exception& exc)
    {
        return message == exc.what();
    }
    catch (...)
    {
    }
    return false;
}

/// Throw an Exception at this location (line < 0: without the line), or rethrow the
/// Exception being handled unchanged if the message is its own (see the file comment).
[[noreturn]] inline void throw_at(const std::string& message, const char* file, int line, const char* func)
{
    if (handles_exception_with_message(message))
        throw;
    if (line < 0)
        throw Exception(message, file, func);
    throw Exception(message, file, line, func);
}
}  // namespace exception_detail

/**
 * @def throw_with_line_number(arg)
 * @brief Convenience macro to throw Exception with automatic location capture.
 *
 * Automatically captures __FILE__, __LINE__, and __func__ at the throw site.
 *
 * @param arg Error message string describing the exception
 *
 * @example
 * @code
 * if (value < 0) {
 *     throw_with_line_number("Value must be non-negative, got: " + std::to_string(value));
 * }
 * @endcode
 */
#define throw_with_line_number(arg) exception_detail::throw_at(arg, __FILE__, __LINE__, __func__);

/**
 * @def throw_without_line_number(arg)
 * @brief Convenience macro to throw Exception without line number.
 *
 * Automatically captures __FILE__ and __func__ at the throw site. In a catch block
 * with the caught message, it rethrows an Exception unchanged (see the file comment).
 *
 * @param arg Error message string describing the exception
 */
#define throw_without_line_number(arg) exception_detail::throw_at(arg, __FILE__, -1, __func__);

#endif
