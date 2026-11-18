/*******************************************************************************
 * WARNING TO AI ASSISTANTS (Claude, ChatGPT, Copilot, etc.):
 * DO NOT MODIFY TEST PARAMETERS WITHOUT EXPLICIT PERMISSION FROM THE USER.
 * - NEVER increase tolerance values (e.g., 1e-7 -> 1e-6)
 * - NEVER decrease field strength or standard deviation values
 * - NEVER change grid sizes, box dimensions, or polymer parameters
 * - NEVER weaken any test conditions to make tests pass
 * These parameters are carefully calibrated. If a test fails, report the
 * failure to the user rather than modifying the test to pass.
 ******************************************************************************/

// The rethrow rule of Exception.h: a catch block that rethrows what it caught with
// the caught message passes an Exception through unchanged (its message names the
// place of the error once) and turns any other exception into an Exception at the
// catch block, which reaches Python as RuntimeError (pybind11 maps std::out_of_range
// to IndexError, so the conversion keeps the Python type of such an error).

#include <iostream>
#include <stdexcept>
#include <string>

#include "Exception.h"

namespace
{
int failures = 0;

void check(bool condition, const std::string& what)
{
    if (!condition)
    {
        std::cout << "FAILED: " << what << std::endl;
        failures++;
    }
}

int count_location_lines(const std::string& message)
{
    int count = 0;
    for (size_t pos = message.find("File: "); pos != std::string::npos; pos = message.find("File: ", pos + 1))
        count++;
    return count;
}

void throw_exception()
{
    throw_with_line_number("the error");
}

void throw_out_of_range()
{
    throw std::out_of_range("index 7 is out of range");
}

// A wrapper of the kind in the computation classes and the bindings
template <typename Body>
void wrap_without_line(Body body)
{
    try
    {
        body();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename Body>
void wrap_with_line(Body body)
{
    try
    {
        body();
    }
    catch (std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}

// What a body throws after passing through the wrappers: (is an Exception, message)
template <typename Body>
std::pair<bool, std::string> outcome(Body body)
{
    try
    {
        body();
    }
    catch (const Exception& exc)
    {
        return {true, exc.what()};
    }
    catch (const std::exception& exc)
    {
        return {false, exc.what()};
    }
    return {false, "nothing thrown"};
}
}  // namespace

int main()
{
    // An Exception keeps its message through nested wrappers of both kinds
    std::string original;
    try
    {
        throw_exception();
    }
    catch (const Exception& exc)
    {
        original = exc.what();
    }
    check(count_location_lines(original) == 1, "the thrown Exception names one location");
    const auto [nested_is_exception, nested] = outcome([] {
        wrap_with_line([] { wrap_without_line([] { wrap_without_line(throw_exception); }); }); });
    std::cout << "Exception through three wrappers:\n" << nested << std::endl;
    check(nested_is_exception, "an Exception stays an Exception");
    check(nested == original, "an Exception passes the wrappers unchanged");

    // A standard exception becomes an Exception at the innermost wrapper, with one location
    const auto [converted_is_exception, converted] = outcome([] {
        wrap_with_line([] { wrap_without_line([] { wrap_without_line(throw_out_of_range); }); }); });
    std::cout << "std::out_of_range through three wrappers:\n" << converted << std::endl;
    check(converted_is_exception, "std::out_of_range becomes an Exception (RuntimeError in Python)");
    check(count_location_lines(converted) == 1, "the converted exception names one location");
    check(converted.find("function <wrap_without_line>\n\tindex 7 is out of range") != std::string::npos,
          "the converted exception keeps the original message after the wrapper's location");

    // Throwing with another message in a catch block is a new Exception
    const auto [new_is_exception, renamed] = outcome([] {
        try
        {
            throw_exception();
        }
        catch (const Exception&)
        {
            throw_with_line_number("a different error");
        }
    });
    check(new_is_exception && renamed.find("a different error") != std::string::npos
          && renamed.find("the error") == std::string::npos, "a new message makes a new Exception");

    // Outside a catch block the macros throw a new Exception
    const auto [plain_is_exception, plain] = outcome([] { throw_without_line_number("plain"); });
    check(plain_is_exception && count_location_lines(plain) == 1 && plain.find("plain") != std::string::npos,
          "a throw outside a catch block names its location");

    if (failures > 0)
        return -1;
    std::cout << "PASSED" << std::endl;
    return 0;
}
