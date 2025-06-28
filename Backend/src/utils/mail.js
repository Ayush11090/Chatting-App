import Mailgen from "mailgen";
import nodemailer from "nodemailer";
import logger from "../logger/winston.logger.js";


/**
 *
 * @param {string} username
 * @param {string} verificationUrl
 * @returns {Mailgen.Content}
 * @description It designs the email verification mail
 */
const emailVerificationMailgenContent = (username, verificationUrl) => {
  return {
    body: {
      name: username,
      intro: "Welcome to our app! We're very excited to have you on board.",
      action: {
        instructions:
          "To verify your email please click on the following button:",
        button: {
          color: "#22BC66", // Optional action button color
          text: "Verify your email",
          link: verificationUrl,
        },
      },
      outro:
        "Need help, or have questions? Just reply to this email, we'd love to help.",
    },
  };
};

/**
 *
 * @param {string} username
 * @param {string} verificationUrl
 * @returns {Mailgen.Content}
 * @description It designs the forgot password mail
 */
const forgotPasswordMailgenContent = (username, passwordResetUrl) => {
  return {
    body: {
      name: username,
      intro: "We got a request to reset the password of our account",
      action: {
        instructions:
          "To reset your password click on the following button or link:",
        button: {
          color: "#22BC66", // Optional action button color
          text: "Reset password",
          link: passwordResetUrl,
        },
      },
      outro:
        "Need help, or have questions? Just reply to this email, we'd love to help.",
    },
  };
};

/**
 *
 * @param {string} username
 * @param {{_id: string, product: Product, quantity: number}[]} items
 * @param {number} totalCost
 * @returns {Mailgen.Content}
 * @description It designs the order creation invoice mail
 */
const orderConfirmationMailgenContent = (username, items, totalCost) => {
  return {
    body: {
      name: username,
      intro: "Your order has been processed successfully.",
      table: {
        data: items?.map((item) => {
          return {
            item: item.product?.name,
            price: "INR " + item.product?.price + "/-",
            quantity: item.quantity,
          };
        }),
        columns: {
          // Optionally, customize the column widths
          customWidth: {
            item: "20%",
            price: "15%",
            quantity: "15%",
          },
          // Optionally, change column text alignment
          customAlignment: {
            price: "right",
            quantity: "right",
          },
        },
      },
      outro: [
        `Total order cost: INR ${totalCost}/-`,
        "You can check the status of your order and more in your order history",
      ],
    },
  };
};

export {
  emailVerificationMailgenContent,
  forgotPasswordMailgenContent,
  orderConfirmationMailgenContent,
};
