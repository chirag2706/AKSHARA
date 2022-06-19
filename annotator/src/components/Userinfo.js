import { useEffect, useState } from "react";
import profile_photo from "./../images/profilePlaceHolder.png";
import { updateUserDetails } from "../scripts/user";
import {connect} from "react-redux";

const UserInfo = (props) => {
    const [editDetails, setEditDetails] = useState(false);
    let userInfo = props.userInfo;
    let setUserInfo = props.setUserInfo;
    const [tempUserInfo, setTempUserInfo] = useState({});

    useEffect(() => {
        setTempUserInfo({
            'name': userInfo.name,
            'description': userInfo.description
        });
    }, [userInfo]);
    
    const handleProfilePhotoChange = props.handleProfilePhotoChange;
    const handleErrorMessage = props.handleErrorMessage;

    const handleOnSave = async () => {
        let status = await updateUserDetails(props.auth.token, {
            'name': tempUserInfo.name,
            'description': tempUserInfo.description
        })
        if(status){
            setUserInfo({ ...userInfo, 'name': tempUserInfo.name, 'description': tempUserInfo.description });
        }
        else {
            handleErrorMessage("User Details not set, Try again");
            setTempUserInfo({
                'name': userInfo.name,
                'description': userInfo.description
            });
        }
    }

    return (
        <div id="user-details-container">
            <div className="container">
                <div className="row justify-content-center mt-4 mb-4">
                    <div
                        className="col-md-3 col-sm-12 p-3 parent"
                        style={{ height: "430px" }}
                    >
                        <div className="child profile-photo-container">
                            <img
                                id="profile-photo"
                                src={ userInfo.image === 'data:image/png;base64, ' ? profile_photo : userInfo.image }
                                className="rounded child profile-photo"
                                alt=""
                                style={{
                                    height: "80%",
                                    width: "100%",
                                    objectFit: "cover",
                                }}
                            />
                            <button
                                className="profile-photo-edit"
                                data-bs-toggle="modal"
                                data-bs-target="#profilePhotoModel"
                            >
                                <i className="fas fa-pen fa-xl"></i>
                            </button>
                            <div
                                className="modal fade"
                                id="profilePhotoModel"
                                tabIndex="-1"
                                aria-labelledby="profilePhotoModelLabel"
                                aria-hidden="true"
                            >
                                <div className="modal-dialog">
                                    <div className="modal-content">
                                        <div className="modal-header">
                                            <h5 className="modal-title">
                                                Select Image
                                            </h5>
                                            <button
                                                id="profile-photo-update-model-close"
                                                type="button"
                                                className="btn-close"
                                                data-bs-dismiss="modal"
                                                aria-label="Close"
                                            ></button>
                                        </div>
                                        <div className="modal-body">
                                            <input
                                                type="file"
                                                id="updated-profile-photo"
                                                name="img"
                                                accept="image/*"
                                            />
                                        </div>
                                        <div className="modal-footer">
                                            <button
                                                type="button"
                                                className="btn btn-secondary"
                                                data-bs-dismiss="modal"
                                            >
                                                Close
                                            </button>
                                            <button
                                                type="button"
                                                className="btn btn-primary"
                                                onClick={async () => {
                                                    setUserInfo({
                                                        ...userInfo,
                                                        image: await handleProfilePhotoChange(),
                                                    });
                                                }}
                                                data-bs-dismiss
                                            >
                                                Save changes
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className="col-md-7 col-sm-12">
                        <div className="p-4">
                            <div className="d-flex justify-content-end">
                                {editDetails === true ? (
                                    <button
                                        type="button"
                                        className="btn btn-outline-success"
                                        onClick={() => {
                                            setEditDetails(false);
                                            handleOnSave();
                                        }}
                                    >
                                        Save
                                    </button>
                                ) : (
                                    <button
                                        type="button"
                                        className="btn btn-outline-dark"
                                        onClick={() => {
                                            setEditDetails(true);
                                        }}
                                    >
                                        Edit
                                    </button>
                                )}
                            </div>
                            <div className="d-grid">
                                <div className="p-2">
                                    <div className="user-field-title">
                                        Username
                                    </div>
                                    <div className="user-field-value">
                                        { userInfo["username"] }
                                    </div>
                                </div>
                                <div className="p-2">
                                    <div className="user-field-title">Name</div>
                                    <div className="user-field-value">
                                        {editDetails === true ? (
                                            <input
                                                type="text"
                                                value={tempUserInfo["name"]}
                                                onChange={(e) => {
                                                    setTempUserInfo({
                                                        ...tempUserInfo,
                                                        name: e.target.value,
                                                    });
                                                }}
                                                style={{
                                                    width: "90%",
                                                    height: "40px",
                                                    marginTop: "10px",
                                                }}
                                            />
                                        ) : (
                                            userInfo["name"]
                                        )}
                                    </div>
                                </div>
                                <div className="p-2">
                                    <div className="user-field-title">
                                        Description
                                    </div>
                                    <div className={"user-field-value description" + ((!editDetails && userInfo["description"] === "") ? ' blank-description' : '')}>
                                        {editDetails === true ? (
                                            <textarea
                                                type="text"
                                                value={tempUserInfo["description"]}
                                                onChange={(e) => {
                                                    setTempUserInfo({
                                                        ...tempUserInfo,
                                                        description:
                                                            e.target.value,
                                                    });
                                                }}
                                                style={{
                                                    width: "90%",
                                                    height: "90px",
                                                    marginTop: "10px",
                                                }}
                                            ></textarea>
                                        ) : (
                                            userInfo["description"]
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};


export default connect((state) => ({ auth: state.auth }))(UserInfo);
