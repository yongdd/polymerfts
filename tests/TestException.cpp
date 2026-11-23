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

// The location line of the throw macros of Exception.h: throw_with_line_number names the
// file, line and function of the throw, throw_without_line_number the file and function,
// once, before the message.

#include <iostream>
#include <string>
#include <utility>

#include "Exception.h"

namespace
{
int failures = 0;
int line_of_throw = 0;

void check(bool condition, const std::string& what)
{
    if (!condition)
    {
        std::cout << "FAILED: " << what << std::endl;
        failures++;
    }
}

void throw_with_line()
{
    line_of_throw = __LINE__ + 1;
    throw_with_line_number("the error");
}

void throw_without_line()
{
    throw_without_line_number("plain");
}

// What a body throws: (is an Exception, message)
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
    const auto [with_is_exception, with_line] = outcome(throw_with_line);
    std::cout << "throw_with_line_number:\n" << with_line << std::endl;
    check(with_is_exception, "throw_with_line_number throws an Exception");
    check(with_line == "File: \"" + std::string(__FILE__) + "\", line: " + std::to_string(line_of_throw)
                       + ", function <throw_with_line>\n\tthe error",
          "throw_with_line_number names the file, line and function of the throw once");

    const auto [without_is_exception, without_line] = outcome(throw_without_line);
    std::cout << "throw_without_line_number:\n" << without_line << std::endl;
    check(without_is_exception, "throw_without_line_number throws an Exception");
    check(without_line == "File: \"" + std::string(__FILE__) + "\", function <throw_without_line>\n\tplain",
          "throw_without_line_number names the file and function of the throw once");

    if (failures > 0)
        return -1;
    std::cout << "PASSED" << std::endl;
    return 0;
}
